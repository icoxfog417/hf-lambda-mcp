import json
from pathlib import Path

import pytest

from hf_lambda_mcp import profile as p

ROOT = Path(__file__).resolve().parents[1]

# Mimics the constructs used by real chat templates (macros with local sets, `is defined` defaults).
TEMPLATE = """
{%- macro check(value, name) -%}{%- set parts = value.split('-') -%}{%- set leftover = parts | length -%}{%- endmacro -%}
{%- if target_lang is not defined -%}{%- set target_lang = 'ja' -%}{%- endif -%}
{{- check(target_lang, 'target_lang') -}}
{%- if source_lang is defined and source_lang is not none -%}{{ source_lang }}{%- endif -%}
{%- if context is defined and context -%}{{ context }}{%- endif -%}
{%- if glossary is defined and glossary -%}{% for t in glossary %}{{ t['source'] }}{% endfor %}{%- endif -%}
{%- if form is defined and form is not none -%}{{ form }}{%- endif -%}
{{ bos_token }}{% for m in messages %}{{ m['content'] }}{% endfor %}
"""


def plamo():
    return p.load_profile(ROOT / "profiles" / "plamo-3-translate.yaml")


def test_template_variables_excludes_internal_and_standard():
    assert p.template_variables(TEMPLATE) == {"target_lang", "source_lang", "context", "glossary", "form"}


def test_profile_params_match_template():
    assert p.check_params_against_template(plamo(), p.template_variables(TEMPLATE)) == []


def test_unknown_param_is_rejected():
    prof = plamo()
    prof["tools"][0]["params"]["tone"] = {"type": "string"}
    with pytest.raises(p.ProfileError, match="tone"):
        p.check_params_against_template(prof, p.template_variables(TEMPLATE))


def test_resolve_source_variants():
    prof = plamo()
    assert p.resolve_source(prof, None)["repo"] == "pfnet/plamo-3-translate-2b-GGUF"
    s = p.resolve_source(prof, "0.8b")
    assert s["file"] == "plamo-3-translate-0.8b-Q4_K_M.gguf"
    with pytest.raises(p.ProfileError):
        p.resolve_source(prof, "70b")


def test_reserved_tool_names():
    prof = plamo()
    prof["tools"][0]["name"] = "start_job"
    with pytest.raises(p.ProfileError):
        p.validate_profile(prof)


def runtime_from(prof):
    return {"name": prof["name"], "tools": prof["tools"], "context_length": 8192}


def test_gateway_schema_shape():
    schema = p.gateway_tool_schema(runtime_from(plamo()))
    names = [t["name"] for t in schema]
    assert names == ["translate", "create_upload", "start_job", "get_job", "get_model_info"]
    tr = schema[0]["inputSchema"]
    assert tr["required"] == ["text", "target_lang"]
    assert tr["properties"]["glossary"]["items"]["properties"]["source"]["type"] == "string"
    assert "required" not in tr["properties"]["target_lang"]
    # schema must be JSON serialisable and small enough for an inline/S3 Gateway schema
    assert len(json.dumps(schema)) < 1_000_000


def test_profile_without_documents_has_only_model_info():
    prof = plamo()
    prof["tools"][0].pop("document")
    names = [t["name"] for t in p.gateway_tool_schema(runtime_from(prof))]
    assert names == ["translate", "get_model_info"]


def test_all_profiles_are_valid():
    for path in (ROOT / "profiles").glob("*.yaml"):
        p.validate_profile(p.load_profile(path))
