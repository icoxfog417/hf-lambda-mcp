import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import mcp.types as types
import pytest

from hf_lambda_mcp.client import cli

STORE: dict[str, bytes] = {}


class Handler(BaseHTTPRequestHandler):
    def do_PUT(self):
        STORE[self.path] = self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200); self.end_headers()

    def do_GET(self):
        body = STORE.get(self.path, b"")
        self.send_response(200 if self.path in STORE else 404)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def http_base():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def text_result(obj, is_error=False):
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(obj))], isError=is_error)


class FakeRemote:
    """Mimics the Gateway: prefixed tool names, JSON text results, async job finishing after 2 polls."""

    def __init__(self, base):
        self.base, self.polls, self.calls = base, 0, []

    async def list_tools(self):
        schema = {"type": "object", "properties": {"text": {"type": "string"}, "target_lang": {"type": "string"}},
                  "required": ["text", "target_lang"]}
        return [types.Tool(name=f"plamo___{n}", description=n, inputSchema=schema if n == "translate" else {"type": "object"})
                for n in ("translate", "create_upload", "start_job", "get_job", "get_model_info")]

    async def call(self, name, args):
        self.calls.append((name, args))
        tool = name.split("___")[1]
        if tool == "get_model_info":
            return text_result({"model": {"repo": "pfnet/x"}, "max_upload_bytes": 1000,
                                "tools": [{"name": "translate", "input": "text", "document": {"formats": ["txt", "md"], "output_name": "{stem}.{target_lang}{ext}"}}]})
        if tool == "translate":
            if not args.get("target_lang"):
                return text_result({"error": "missing target_lang"})
            return text_result({"output": args["text"].upper()})
        if tool == "create_upload":
            return text_result({"upload_url": f"{self.base}/up/{args['filename']}", "key": f"uploads/1/{args['filename']}"})
        if tool == "start_job":
            self.job_args = args
            return text_result({"job_id": "a" * 32, "status": "queued"})
        if tool == "get_job":
            self.polls += 1
            if self.polls < 3:
                return text_result({"status": "running", "progress": {"done": self.polls, "total": 3}})
            name = self.job_args["key"].split("/")[-1]
            STORE["/result"] = STORE[f"/up/{name}"].upper()
            stem, ext = name.rsplit(".", 1)
            return text_result({"status": "succeeded", "output_name": f"{stem}.{self.job_args['arguments']['target_lang']}.{ext}",
                                "download_url": f"{self.base}/result", "stats": {"chunks": 3}})
        raise AssertionError(name)

    async def call_json(self, name, args):
        return await cli.Remote.call_json(self, name, args)


@pytest.fixture
def bridge(http_base, monkeypatch):
    monkeypatch.setattr(cli, "POLL_INTERVAL_S", 0.01)
    return cli.Bridge(FakeRemote(http_base), expose_job_tools=False, wait_s=30)


def run(coro):
    return asyncio.run(coro)


def test_tools_are_unprefixed_and_file_variant_generated(bridge):
    tools = {t.name: t for t in run(bridge.load())}
    assert set(tools) == {"translate", "get_model_info", "translate_file", "download_job_result"}
    f = tools["translate_file"].inputSchema
    assert f["required"] == ["input_path", "target_lang"]
    assert "text" not in f["properties"] and "output_path" in f["properties"]


def test_proxy_call_and_error_unwrapping(bridge):
    r = run(bridge.call("translate", {"text": "hi", "target_lang": "en"}, None))
    assert not r.isError and json.loads(r.content[0].text)["output"] == "HI"
    r = run(bridge.call("translate", {"text": "hi"}, None))
    assert r.isError and "missing target_lang" in r.content[0].text


def test_file_tool_roundtrip(bridge, tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("# hello\n\nworld\n", encoding="utf-8")
    progress = []

    async def prog(done, total):
        progress.append((done, total))

    r = run(bridge.call("translate_file", {"input_path": str(src), "target_lang": "en"}, prog))
    assert not r.isError, r.content[0].text
    out = json.loads(r.content[0].text)
    assert out["output_path"] == str(tmp_path / "notes.en.md")
    assert (tmp_path / "notes.en.md").read_text() == "# HELLO\n\nWORLD\n"
    assert progress and progress[0] == (1.0, 3.0)
    # job arguments exclude local-only parameters
    assert bridge.remote.job_args["arguments"] == {"target_lang": "en"}


def test_file_tool_explicit_output_and_validation(bridge, tmp_path):
    src = tmp_path / "a.txt"
    src.write_text("abc")
    out = tmp_path / "sub" / "b.txt"
    r = run(bridge.call("translate_file", {"input_path": str(src), "output_path": str(out), "target_lang": "ja"}, None))
    assert not r.isError and out.read_text() == "ABC"
    assert run(bridge.call("translate_file", {"input_path": str(tmp_path / "nope.md"), "target_lang": "en"}, None)).isError
    (tmp_path / "x.pdf").write_bytes(b"%PDF")
    assert "unsupported" in run(bridge.call("translate_file", {"input_path": str(tmp_path / "x.pdf"), "target_lang": "en"}, None)).content[0].text
    (tmp_path / "big.md").write_text("x" * 2000)
    assert "too large" in run(bridge.call("translate_file", {"input_path": str(tmp_path / "big.md"), "target_lang": "en"}, None)).content[0].text


def test_hidden_job_tools_are_not_callable(bridge):
    assert run(bridge.call("start_job", {}, None)).isError


def test_region_from_endpoint():
    assert cli.region_from_endpoint("https://g-abc.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp") == "us-east-2"
    assert cli.region_from_endpoint("https://example.com/mcp") is None
