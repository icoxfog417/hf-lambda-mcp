"""AWS Lambda entry point (AgentCore Gateway Lambda target).

Module import (Lambda init) starts llama-server and runs a warm-up generation so that with
SnapStart the loaded model is captured in the snapshot. Each invocation is either
- a Gateway tool call: tool name in ``context.client_context.custom['bedrockAgentCoreToolName']``
  (``<target>___<tool>``) and the tool arguments as the event, or
- an asynchronous document-job worker invocation: ``{"hf_lambda_mcp_job": {...}}``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from . import jobs
from .engine import Engine, ToolInputError
from .llama import LlamaServer, ModelError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
with open(os.path.join(MODEL_DIR, "runtime.json"), encoding="utf-8") as _f:
    RUNTIME: dict[str, Any] = json.load(_f)

SERVER = LlamaServer(RUNTIME)
_t0 = time.time()
_load_s = SERVER.start()
ENGINE = Engine(RUNTIME, SERVER)
ENGINE.warmup()
INIT_STATS = {"prefetch_s": round(SERVER.prefetch_s, 2), "load_s": round(_load_s, 2), "init_s": round(time.time() - _t0, 2), "cpus": os.cpu_count()}
logger.info("model ready %s", json.dumps(INIT_STATS))


def _tool_name(event: dict[str, Any], context: Any) -> str:
    custom = getattr(getattr(context, "client_context", None), "custom", None) or {}
    name = custom.get("bedrockAgentCoreToolName") or event.pop("__tool__", "")
    return name.split("___", 1)[1] if "___" in name else name


def model_info() -> dict[str, Any]:
    m = RUNTIME["model"]
    return {
        "profile": RUNTIME["name"],
        "model": {k: m.get(k) for k in ("repo", "file", "variant", "revision", "architecture")},
        "license": {k: m.get(k) for k in ("license", "license_name", "license_link")},
        "license_notice": "Check the model license before use; this server only hosts the model.",
        "context_length": RUNTIME["context_length"],
        "tools": [
            {"name": t["name"], "input": t["input"]["name"], "params": sorted(t.get("params", {})),
             "required": sorted(k for k, v in t.get("params", {}).items() if v.get("required")),
             **({"document": {k: t["document"][k] for k in ("formats", "output_name") if k in t["document"]}} if t.get("document") else {})}
            for t in RUNTIME["tools"]
        ],
        "max_upload_bytes": jobs.MAX_UPLOAD_BYTES,
        "runtime": INIT_STATS,
    }


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    event = dict(event or {})
    if "hf_lambda_mcp_job" in event:
        return jobs.run_worker(ENGINE, event["hf_lambda_mcp_job"], context)

    name = _tool_name(event, context)
    t0 = time.time()
    try:
        if name == "get_model_info":
            result = model_info()
        elif name == "create_upload":
            result = jobs.create_upload(ENGINE, event)
        elif name == "start_job":
            result = jobs.start_job(ENGINE, event, context)
        elif name == "get_job":
            result = jobs.get_job(ENGINE, event)
        else:
            result = ENGINE.run_inline(name, event)
    except (ToolInputError, ModelError) as e:
        logger.info("tool=%s rejected: %s", name, e)
        return {"error": str(e)}
    logger.info("tool=%s elapsed=%.2fs", name, time.time() - t0)
    return result
