"""Local stdio MCP server that fronts an hf-lambda-mcp AgentCore Gateway.

- Proxies the remote tools (SigV4 via mcp-proxy-for-aws), stripping the ``<target>___`` prefix.
- For every remote tool with document support, adds a local ``<tool>_file`` tool that reads a
  local file, uploads it through a presigned URL, runs an asynchronous job, and writes the result
  next to the input. File contents never pass through the agent's context window.

Nothing here is model specific: capabilities are discovered from the remote ``get_model_info``.

Usage (e.g. in .mcp.json):

    uvx --from git+https://github.com/icoxfog417/hf-lambda-mcp hf-lambda-mcp <MCP endpoint URL>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import mcp.types as types
from mcp import ClientSession
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger("hf_lambda_mcp.client")

JOB_TOOLS = {"create_upload", "start_job", "get_job"}
FILE_SUFFIX = "_file"
DOWNLOAD_TOOL = "download_job_result"
POLL_INTERVAL_S = 3.0


def region_from_endpoint(endpoint: str) -> str | None:
    m = re.search(r"\.gateway\.bedrock-agentcore\.([a-z0-9-]+)\.amazonaws\.com", endpoint)
    return m.group(1) if m else None


class ToolError(Exception):
    pass


_current_session: contextvars.ContextVar[ClientSession | None] = contextvars.ContextVar("hf_lambda_mcp_session", default=None)


class Remote:
    """MCP client for the Gateway.

    A session is opened per local tool call with ``async with remote.session():`` and used by the
    methods below through a context variable. Entering and exiting the transport in the same task is
    required by anyio, and short-lived sessions always sign with fresh credentials.
    """

    def __init__(self, endpoint: str, region: str | None, profile: str | None, timeout: float):
        self.endpoint, self.region, self.profile, self.timeout = endpoint, region, profile, timeout

    @contextlib.asynccontextmanager
    async def session(self):
        if _current_session.get() is not None:  # nested use within one call
            yield
            return
        from mcp_proxy_for_aws.client import aws_iam_streamablehttp_client

        async with aws_iam_streamablehttp_client(
            endpoint=self.endpoint, aws_service="bedrock-agentcore", aws_region=self.region,
            aws_profile=self.profile, timeout=self.timeout, sse_read_timeout=self.timeout,
        ) as (read, write, _), ClientSession(read, write) as s:
            await s.initialize()
            token = _current_session.set(s)
            try:
                yield
            finally:
                _current_session.reset(token)

    async def _with_session(self, fn):
        s = _current_session.get()
        if s is None:
            async with self.session():
                return await fn(_current_session.get())
        return await fn(s)

    async def close(self) -> None:
        return None

    async def list_tools(self) -> list[types.Tool]:
        async def fn(s: ClientSession):
            tools, cursor = [], None
            while True:
                r = await s.list_tools(cursor=cursor) if cursor else await s.list_tools()
                tools += r.tools
                cursor = r.nextCursor
                if not cursor:
                    return tools
        return await self._with_session(fn)

    async def call(self, name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        return await self._with_session(lambda s: s.call_tool(name, arguments))

    async def call_json(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        r = await self.call(name, arguments)
        text = "".join(c.text for c in r.content if isinstance(c, types.TextContent))
        if r.isError:
            raise ToolError(text or f"remote tool {name} failed")
        data = r.structuredContent if isinstance(r.structuredContent, dict) and r.structuredContent else None
        if data is None:
            try:
                data = json.loads(text)
            except ValueError:
                raise ToolError(f"unexpected response from {name}: {text[:500]}") from None
        if isinstance(data, dict) and "error" in data and len(data) == 1:
            raise ToolError(str(data["error"]))
        return data


class Bridge:
    def __init__(self, remote: Remote, expose_job_tools: bool, wait_s: float):
        self.remote = remote
        self.expose_job_tools = expose_job_tools
        self.wait_s = wait_s
        self.remote_names: dict[str, str] = {}  # local name -> remote (prefixed) name
        self.tools: list[types.Tool] | None = None
        self.info: dict[str, Any] = {}
        self.doc_tools: dict[str, dict[str, Any]] = {}

    async def load(self) -> list[types.Tool]:
        if self.tools is not None:
            return self.tools
        async with self.remote.session():
            return await self._load()

    async def _load(self) -> list[types.Tool]:
        remote_tools = await self.remote.list_tools()
        by_local: dict[str, types.Tool] = {}
        for t in remote_tools:
            local = t.name.split("___", 1)[1] if "___" in t.name else t.name
            self.remote_names[local] = t.name
            by_local[local] = t
        if "get_model_info" in self.remote_names:
            self.info = await self.remote.call_json(self.remote_names["get_model_info"], {})
        exposed: list[types.Tool] = []
        for local, t in by_local.items():
            if local in JOB_TOOLS and not self.expose_job_tools:
                continue
            exposed.append(t.model_copy(update={"name": local}))
        has_jobs = JOB_TOOLS <= set(self.remote_names)
        for spec in self.info.get("tools", []):
            if spec.get("document") and spec["name"] in by_local and has_jobs:
                self.doc_tools[spec["name"] + FILE_SUFFIX] = spec
                exposed.append(self._file_tool(by_local[spec["name"]], spec))
        if self.doc_tools:
            exposed.append(types.Tool(
                name=DOWNLOAD_TOOL,
                description="Wait for (or download) the result of a file job that was still running when a *_file tool returned.",
                inputSchema={"type": "object", "properties": {
                    "job_id": {"type": "string", "description": "job_id returned by the *_file tool"},
                    "output_path": {"type": "string", "description": "Where to write the result file"},
                }, "required": ["job_id", "output_path"]},
            ))
        self.tools = exposed
        return exposed

    def _file_tool(self, tool: types.Tool, spec: dict[str, Any]) -> types.Tool:
        schema = json.loads(json.dumps(tool.inputSchema))
        props: dict[str, Any] = schema.get("properties", {})
        props.pop(spec["input"], None)
        doc = spec["document"]
        formats = ", ".join(f".{f}" for f in doc.get("formats", []))
        new_props = {
            "input_path": {"type": "string", "description": f"Path of the local input file ({formats})."},
            "output_path": {
                "type": "string",
                "description": f"Output file or directory. Default: next to the input, named '{doc.get('output_name')}'.",
            },
        }
        new_props.update(props)
        required = ["input_path"] + [r for r in schema.get("required", []) if r != spec["input"]]
        description = (
            f"File variant of '{spec['name']}': reads a local {formats} file, processes it remotely chunk by chunk "
            "while preserving document structure, and writes the result to a local file. Returns only the output "
            "path and statistics, so the file content never enters the conversation. "
            f"Model: {self.info.get('model', {}).get('repo', '')}.\n\nOriginal tool description: {tool.description or ''}"
        )
        return types.Tool(name=spec["name"] + FILE_SUFFIX, description=description,
                          inputSchema={"type": "object", "properties": new_props, "required": required})

    # ------------------------------------------------------------------ calls

    async def call(self, name: str, arguments: dict[str, Any], progress: Any) -> types.CallToolResult:
        try:
            async with self.remote.session():
                return await self._call(name, arguments, progress)
        except ToolError as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(e))], isError=True)
        except Exception as e:  # noqa: BLE001 - report transport/auth failures as tool errors
            logger.exception("tool call failed")
            return types.CallToolResult(content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")], isError=True)

    async def _call(self, name: str, arguments: dict[str, Any], progress: Any) -> types.CallToolResult:
        await self.load()
        try:
            if name in self.doc_tools:
                return _ok(await self._run_file(self.doc_tools[name], arguments, progress))
            if name == DOWNLOAD_TOOL and self.doc_tools:
                out = Path(arguments["output_path"]).expanduser()
                return _ok(await self._wait_and_download(arguments["job_id"], out, progress, time.time()))
            if name not in self.remote_names or (name in JOB_TOOLS and not self.expose_job_tools):
                raise ToolError(f"unknown tool: {name}")
            r = await self.remote.call(self.remote_names[name], arguments)
            return _unwrap_error(r)
        except ToolError as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(e))], isError=True)

    async def _run_file(self, spec: dict[str, Any], args: dict[str, Any], progress: Any) -> dict[str, Any]:
        args = dict(args)
        src = Path(str(args.pop("input_path", ""))).expanduser().resolve()
        output = args.pop("output_path", None)
        if not src.is_file():
            raise ToolError(f"input file not found: {src}")
        fmt = src.suffix.lower().lstrip(".")
        fmt = {"markdown": "md", "text": "txt"}.get(fmt, fmt)
        formats = spec["document"].get("formats", [])
        if fmt not in formats:
            raise ToolError(f"unsupported file type '{src.suffix}'; supported: {formats}")
        data = src.read_bytes()
        limit = int(self.info.get("max_upload_bytes") or 0)
        if limit and len(data) > limit:
            raise ToolError(f"file is too large ({len(data)} bytes > {limit})")
        try:
            data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ToolError("input file must be UTF-8 text") from None

        upload = await self.remote.call_json(self.remote_names["create_upload"], {"filename": src.name})
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.put(upload["upload_url"], content=data)
            if r.status_code >= 300:
                raise ToolError(f"upload failed: HTTP {r.status_code} {r.text[:300]}")
        job = await self.remote.call_json(
            self.remote_names["start_job"], {"tool": spec["name"], "key": upload["key"], "arguments": args}
        )
        out = Path(output).expanduser() if output else src.parent
        return await self._wait_and_download(job["job_id"], out, progress, time.time(), source=src)

    async def _wait_and_download(self, job_id: str, out: Path, progress: Any, started: float,
                                 source: Path | None = None) -> dict[str, Any]:
        while True:
            status = await self.remote.call_json(self.remote_names["get_job"], {"job_id": job_id})
            p = status.get("progress") or {}
            if progress is not None and p.get("total"):
                await progress(float(p.get("done", 0)), float(p["total"]))
            if status["status"] == "succeeded":
                break
            if status["status"] == "failed":
                raise ToolError(f"job {job_id} failed: {status.get('error')}")
            if time.time() - started > self.wait_s:
                return {
                    "job_id": job_id, "status": status["status"], "progress": p,
                    "note": f"Still running. Call {DOWNLOAD_TOOL} with this job_id and output_path to wait for the result.",
                    "output_path": str(out / status.get("output_name", "")) if out.is_dir() else str(out),
                }
            await asyncio.sleep(POLL_INTERVAL_S)

        target = out / status["output_name"] if out.is_dir() else out
        if source is not None and target.resolve() == source:
            raise ToolError("output_path must differ from input_path")
        target.parent.mkdir(parents=True, exist_ok=True)
        async with httpx.AsyncClient(timeout=120) as http:
            r = await http.get(status["download_url"])
            if r.status_code >= 300:
                raise ToolError(f"download failed: HTTP {r.status_code}")
            target.write_bytes(r.content)
        return {"output_path": str(target), "job_id": job_id, "bytes": len(r.content), **(status.get("stats") or {})}


def _ok(data: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))])


def _unwrap_error(r: types.CallToolResult) -> types.CallToolResult:
    """Lambda tools report input errors as {"error": "..."}; surface them as MCP tool errors."""
    if r.isError:
        return r
    text = "".join(c.text for c in r.content if isinstance(c, types.TextContent))
    with contextlib.suppress(ValueError, TypeError):
        data = json.loads(text)
        if isinstance(data, dict) and set(data) == {"error"}:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(data["error"]))], isError=True)
    return r


def build_server(bridge: Bridge) -> Server:
    server: Server = Server("hf-lambda-mcp")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return await bridge.load()

    @server.call_tool(validate_input=False)
    async def call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        ctx = server.request_context
        token = ctx.meta.progressToken if ctx.meta else None
        progress = None
        if token is not None:
            async def progress(done: float, total: float) -> None:
                await ctx.session.send_progress_notification(token, done, total)
        return await bridge.call(name, arguments or {}, progress)

    return server


async def _amain(args: argparse.Namespace) -> None:
    region = args.region or region_from_endpoint(args.endpoint)
    remote = Remote(args.endpoint, region, args.profile, args.timeout)
    bridge = Bridge(remote, args.expose_job_tools, args.wait)
    server = build_server(bridge)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await remote.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hf-lambda-mcp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("endpoint", help="AgentCore Gateway MCP endpoint URL (stack output McpEndpointUrl)")
    p.add_argument("--region", help="AWS region (default: parsed from the endpoint URL)")
    p.add_argument("--profile", help="AWS profile name")
    p.add_argument("--timeout", type=float, default=900, help="Remote request timeout in seconds")
    p.add_argument("--wait", type=float, default=600, help="Max seconds a *_file tool waits before returning a job_id")
    p.add_argument("--expose-job-tools", action="store_true", help="Also expose create_upload/start_job/get_job")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr)
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
