"""Model profiles: load YAML, resolve against Hugging Face artifacts, and emit build outputs.

Build outputs (written by ``hf-lambda-mcp-build``):

- ``runtime.json``      resolved configuration baked into the Lambda image
- ``tool-schema.json``  AgentCore Gateway Lambda target tool schema
- model file + chat template downloaded from Hugging Face

The runtime (Lambda) only reads ``runtime.json`` and therefore needs no third-party packages.
Build-time helpers import ``yaml``, ``jinja2``, ``gguf`` and ``huggingface_hub`` lazily.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from . import documents

DEFAULT_CONTEXT_LENGTH = 8192
DEFAULT_MAX_CHUNK_TOKENS = 1024

# Variables every chat template may reference; these are never exposed as tool parameters.
STANDARD_TEMPLATE_VARS = {
    "messages", "tools", "documents", "add_generation_prompt", "bos_token", "eos_token",
    "raise_exception", "strftime_now", "date_string", "tools_in_user_message", "builtin_tools",
    "custom_tools", "enable_thinking", "chat_template_kwargs", "namespace", "range", "loop",
}

# Generic (model-independent) remote tools for asynchronous document jobs.
GENERIC_TOOLS = ("create_upload", "start_job", "get_job", "get_model_info")


class ProfileError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# Loading and resolving
# --------------------------------------------------------------------------------------------


def load_profile(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        profile = yaml.safe_load(f)
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    for key in ("name", "source", "tools"):
        if key not in profile:
            raise ProfileError(f"profile is missing '{key}'")
    src = profile["source"]
    for key in ("repo", "file"):
        if key not in src:
            raise ProfileError(f"source is missing '{key}'")
    names = set()
    for tool in profile["tools"]:
        name = tool.get("name")
        if not name or not name.replace("_", "").isalnum():
            raise ProfileError(f"invalid tool name: {name!r}")
        if name in GENERIC_TOOLS or name in names:
            raise ProfileError(f"tool name '{name}' is reserved or duplicated")
        names.add(name)
        if "description" not in tool or "input" not in tool:
            raise ProfileError(f"tool '{name}' needs 'description' and 'input'")
        doc = tool.get("document")
        if doc:
            bad = set(doc.get("formats", [])) - set(documents.SUPPORTED_FORMATS)
            if bad:
                raise ProfileError(f"tool '{name}': unsupported document formats {sorted(bad)}")


def resolve_source(profile: dict[str, Any], variant: str | None) -> dict[str, str]:
    src = profile["source"]
    variants = src.get("variants")
    variant = variant or src.get("default_variant")
    if variants:
        if variant not in variants:
            raise ProfileError(f"variant '{variant}' not in {variants}")
    elif "{variant}" in src["repo"] + src["file"]:
        raise ProfileError("profile uses {variant} but declares no variants")
    fmt = {"variant": variant or ""}
    return {"variant": variant or "", "repo": src["repo"].format(**fmt), "file": src["file"].format(**fmt)}


def template_variables(template: str) -> set[str]:
    """Names referenced by a chat template, minus the standard ones."""
    import jinja2
    from jinja2 import meta

    env = jinja2.Environment()
    ast = env.parse(template)
    names = set(meta.find_undeclared_variables(ast))
    # Variables assigned anywhere (e.g. `{% set %}` inside macros) are template-internal ...
    assigned = {n.name for node in ast.find_all(jinja2.nodes.Assign) for n in node.target.find_all(jinja2.nodes.Name)}
    names -= assigned
    # ... unless the template checks `x is defined`, which marks x as a caller-supplied kwarg.
    for node in ast.find_all(jinja2.nodes.Test):
        if node.name == "defined" and isinstance(node.node, jinja2.nodes.Name):
            names.add(node.node.name)
    return names - STANDARD_TEMPLATE_VARS


def check_params_against_template(profile: dict[str, Any], template_vars: set[str]) -> list[str]:
    """Raise if a profile parameter is unknown to the template; return template vars not exposed."""
    exposed: set[str] = set()
    for tool in profile["tools"]:
        for name in tool.get("params", {}):
            if name not in template_vars:
                raise ProfileError(
                    f"tool '{tool['name']}': parameter '{name}' is not used by the chat template "
                    f"(template variables: {sorted(template_vars)})"
                )
            exposed.add(name)
    return sorted(template_vars - exposed)


# --------------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------------


def _json_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Convert a profile param spec to JSON schema (drops profile-only keys)."""
    out = {k: copy.deepcopy(v) for k, v in spec.items() if k not in ("required",)}
    out.setdefault("type", "string")
    if out["type"] == "array" and "items" in out:
        out["items"] = _json_schema(out["items"]) if isinstance(out["items"], dict) else out["items"]
    if out["type"] == "object" and "properties" in out:
        out["properties"] = {k: _json_schema(v) for k, v in out["properties"].items()}
    return out


def param_schema(tool: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    props = {name: _json_schema(spec) for name, spec in tool.get("params", {}).items()}
    required = [name for name, spec in tool.get("params", {}).items() if spec.get("required")]
    return props, required


def content_tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    props, required = param_schema(tool)
    inp = tool["input"]
    schema_props = {inp["name"]: {"type": "string", "description": inp.get("description", "Input text.")}}
    schema_props.update(props)
    return {
        "name": tool["name"],
        "description": tool["description"].strip(),
        "inputSchema": {"type": "object", "properties": schema_props, "required": [inp["name"], *required]},
    }


def document_tools(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in runtime["tools"] if t.get("document")]


def generic_tool_schemas(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    doc_tools = document_tools(runtime)
    if not doc_tools:
        return [_model_info_schema()]
    summary = "; ".join(f"{t['name']} ({', '.join(t['document']['formats'])})" for t in doc_tools)
    return [
        {
            "name": "create_upload",
            "description": (
                "Create a short-lived presigned URL for uploading a document that will be processed by "
                "start_job. HTTP PUT the raw file bytes to upload_url, then call start_job with the returned key. "
                f"Document-capable tools: {summary}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "File name including extension, e.g. report.md"}},
                "required": ["filename"],
            },
        },
        {
            "name": "start_job",
            "description": (
                "Start an asynchronous job that runs a tool over an uploaded document chunk by chunk, "
                "preserving document structure. Returns job_id; poll get_job until status is 'succeeded'. "
                f"Document-capable tools: {summary}."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "Tool to run, e.g. " + doc_tools[0]["name"]},
                    "key": {"type": "string", "description": "Object key returned by create_upload"},
                    "arguments": {
                        "type": "object",
                        "description": "Arguments for the tool, excluding the text input (e.g. {\"target_lang\": \"en\"})",
                    },
                },
                "required": ["tool", "key", "arguments"],
            },
        },
        {
            "name": "get_job",
            "description": (
                "Get the status and progress of a document job. When status is 'succeeded', download_url "
                "is a presigned URL for the result file (HTTP GET)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Job ID returned by start_job"}},
                "required": ["job_id"],
            },
        },
        _model_info_schema(),
    ]


def _model_info_schema() -> dict[str, Any]:
    return {
        "name": "get_model_info",
        "description": (
            "Describe the model behind this server: Hugging Face repository, model file, license, "
            "tools and their document capabilities."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    }


def gateway_tool_schema(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [content_tool_schema(t) for t in runtime["tools"]] + generic_tool_schemas(runtime)


# --------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------


def read_gguf_metadata(path: str | Path) -> dict[str, Any]:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    def value(key: str) -> Any:
        field = reader.fields.get(key)
        if field is None:
            return None
        return field.contents()

    arch = value("general.architecture")
    return {
        "architecture": arch,
        "name": value("general.name"),
        "context_length": value(f"{arch}.context_length"),
        "chat_template": value("tokenizer.chat_template"),
    }


def build(profile_path: str, variant: str | None, out_dir: str, context_length: int | None = None,
          download: bool = True) -> dict[str, Any]:
    from huggingface_hub import HfApi, hf_hub_download

    profile = load_profile(profile_path)
    source = resolve_source(profile, variant)
    out = Path(out_dir)
    (out / "model").mkdir(parents=True, exist_ok=True)

    api = HfApi()
    info = api.model_info(source["repo"])
    siblings = {s.rfilename for s in (info.siblings or [])}
    if source["file"] not in siblings:
        raise ProfileError(f"{source['file']} not found in {source['repo']}; files: {sorted(siblings)}")
    card = info.card_data.to_dict() if info.card_data else {}

    model_path = out / "model" / "model.gguf"
    if download:
        downloaded = Path(hf_hub_download(source["repo"], source["file"], revision=info.sha, local_dir=out / "hf"))
        downloaded.replace(model_path)
        shutil.rmtree(out / "hf", ignore_errors=True)
    gguf_meta = read_gguf_metadata(model_path) if model_path.exists() else {}

    template_file = None
    if "chat_template.jinja" in siblings:
        template = Path(hf_hub_download(source["repo"], "chat_template.jinja", revision=info.sha)).read_text(encoding="utf-8")
        template_file = "chat_template.jinja"
        (out / "model" / template_file).write_text(template, encoding="utf-8")
    elif gguf_meta.get("chat_template"):
        template = gguf_meta["chat_template"]
    else:
        raise ProfileError("no chat template found (neither chat_template.jinja nor GGUF metadata)")

    tvars = template_variables(template)
    unexposed = check_params_against_template(profile, tvars)
    if unexposed:
        print(f"note: chat template variables not exposed as tool parameters: {unexposed}", file=sys.stderr)

    model_ctx = int(gguf_meta.get("context_length") or DEFAULT_CONTEXT_LENGTH)
    ctx = int(context_length or profile.get("runtime", {}).get("context_length") or min(model_ctx, DEFAULT_CONTEXT_LENGTH))

    tools = []
    for tool in profile["tools"]:
        t = copy.deepcopy(tool)
        if t.get("document"):
            t["document"].setdefault("max_chunk_tokens", DEFAULT_MAX_CHUNK_TOKENS)
            t["document"].setdefault("history", 0)
            t["document"].setdefault("output_name", "{stem}.out{ext}")
        tools.append(t)

    runtime = {
        "name": profile["name"],
        "model": {
            "repo": source["repo"],
            "file": source["file"],
            "variant": source["variant"],
            "revision": info.sha,
            "architecture": gguf_meta.get("architecture"),
            "context_length": model_ctx,
            "license": card.get("license"),
            "license_name": card.get("license_name"),
            "license_link": card.get("license_link") or f"https://huggingface.co/{source['repo']}",
            "chat_template_file": template_file,
        },
        "context_length": ctx,
        "generation": profile.get("generation", {}),
        "stop": profile.get("stop"),
        "template_variables": sorted(tvars),
        "tools": tools,
    }
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "tool-schema.json").write_text(json.dumps(gateway_tool_schema(runtime), indent=2, ensure_ascii=False), encoding="utf-8")
    return runtime


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Resolve a model profile and download its artifacts for the Lambda image")
    p.add_argument("profile", help="Path to profile YAML (e.g. profiles/plamo-3-translate.yaml)")
    p.add_argument("--variant", default=None)
    p.add_argument("--out", default="build")
    p.add_argument("--context-length", type=int, default=None)
    p.add_argument("--no-download", action="store_true", help="Resolve metadata and schemas only")
    args = p.parse_args(argv)
    runtime = build(args.profile, args.variant, args.out, args.context_length, download=not args.no_download)
    print(json.dumps({"model": runtime["model"], "context_length": runtime["context_length"]}, indent=2))


if __name__ == "__main__":
    main()
