import copy
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


class FakeServer:
    """Stands in for llama-server: 'translates' by upper-casing the last user message."""

    def __init__(self):
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def chat(self, messages, template_kwargs, max_tokens=None):
        self.calls.append({"messages": messages, "kwargs": template_kwargs, "max_tokens": max_tokens})
        text = messages[-1]["content"]
        return {"text": text.upper(), "finish_reason": "stop", "prompt_tokens": sum(len(m["content"].split()) for m in messages),
                "completion_tokens": len(text.split()), "timings": {}}

    def _request(self, *a, **k):
        return {}


@pytest.fixture
def runtime():
    prof = yaml.safe_load((ROOT / "profiles" / "plamo-3-translate.yaml").read_text())
    tools = copy.deepcopy(prof["tools"])
    for t in tools:
        t["document"]["max_chunk_tokens"] = 12
    return {"name": prof["name"], "model": {"repo": "x/y", "file": "m.gguf"}, "context_length": 64,
            "generation": {}, "tools": tools}


@pytest.fixture
def engine(runtime):
    from hf_lambda_mcp.server.engine import Engine

    return Engine(runtime, FakeServer())
