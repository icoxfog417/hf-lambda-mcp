"""Manage a local llama-server process and call its OpenAI-compatible API.

The process is started during Lambda init so that, with SnapStart, the loaded model is part
of the snapshot. Only the standard library is used.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

LLAMA_BIN = os.environ.get("LLAMA_SERVER_BIN", "/opt/llama/llama-server")
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/model")
PORT = int(os.environ.get("LLAMA_PORT", "18080"))
LOG_PATH = "/tmp/llama-server.log"


class ModelError(RuntimeError):
    """Raised for errors reported by the model server (e.g. chat template validation)."""


class LlamaServer:
    def __init__(self, runtime: dict[str, Any]):
        self.runtime = runtime
        self.proc: subprocess.Popen | None = None
        self.prefetch_s = 0.0

    # ------------------------------------------------------------------ lifecycle

    def command(self) -> list[str]:
        threads = str(int(os.environ.get("LLAMA_THREADS", "0")) or os.cpu_count() or 2)
        cmd = [
            LLAMA_BIN,
            "--model", f"{MODEL_DIR}/model.gguf",
            "--host", "127.0.0.1", "--port", str(PORT),
            "--ctx-size", str(self.runtime["context_length"]),
            "--threads", threads, "--threads-batch", threads,
            "--parallel", "1",
            "--jinja",
            "--no-webui",
        ]
        tpl = self.runtime["model"].get("chat_template_file")
        if tpl:
            cmd += ["--chat-template-file", f"{MODEL_DIR}/{tpl}"]
        extra = os.environ.get("LLAMA_EXTRA_ARGS")
        if extra:
            cmd += extra.split()
        return cmd

    def prefetch(self) -> float:
        """Read the model file sequentially with parallel threads before llama-server mmaps it.

        Lambda loads container images lazily; llama.cpp's scattered mmap page faults against a lazy image
        take minutes for multi-GB models, while parallel sequential reads warm the page cache in seconds.
        """
        from concurrent.futures import ThreadPoolExecutor

        threads = int(os.environ.get("PREFETCH_THREADS", "8"))
        if threads <= 0:
            return 0.0
        t0 = time.time()
        path = f"{MODEL_DIR}/model.gguf"
        size = os.path.getsize(path)
        block = 16 << 20

        def read(i: int) -> None:
            fd = os.open(path, os.O_RDONLY)
            try:
                offset = i * block
                while offset < size:
                    os.pread(fd, block, offset)
                    offset += threads * block
            finally:
                os.close(fd)

        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(read, range(threads)))
        return time.time() - t0

    def start(self, timeout: float = 600) -> float:
        t0 = time.time()
        self.prefetch_s = self.prefetch()
        log = open(LOG_PATH, "w")
        self.proc = subprocess.Popen(self.command(), stdout=log, stderr=subprocess.STDOUT)
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode}): {self.log_tail()}")
            try:
                if self._request("GET", "/health", timeout=2).get("status") == "ok":
                    return time.time() - t0
            except (OSError, ModelError):
                pass
            time.sleep(0.1)
        raise RuntimeError(f"llama-server did not become healthy in {timeout}s: {self.log_tail()}")

    def ensure_running(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            logger.warning("llama-server not running; restarting")
            self.start()

    def log_tail(self, n: int = 4000) -> str:
        try:
            with open(LOG_PATH) as f:
                return f.read()[-n:]
        except OSError:
            return ""

    # ------------------------------------------------------------------ API

    def _request(self, method: str, path: str, body: Any = None, timeout: float = 900) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(raw).get("error", {})
                msg = msg.get("message", raw) if isinstance(msg, dict) else str(msg)
            except ValueError:
                msg = raw
            raise ModelError(msg) from None

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._request("POST", "/tokenize", {"content": text, "add_special": False})["tokens"])

    def chat(self, messages: list[dict[str, str]], template_kwargs: dict[str, Any], max_tokens: int | None = None) -> dict[str, Any]:
        self.ensure_running()
        body: dict[str, Any] = {"messages": messages, "chat_template_kwargs": template_kwargs, "cache_prompt": True}
        body.update(self.runtime.get("generation") or {})
        if self.runtime.get("stop"):
            body["stop"] = self.runtime["stop"]
        if max_tokens:
            body["max_tokens"] = max_tokens
        r = self._request("POST", "/v1/chat/completions", body)
        choice = r["choices"][0]
        return {
            "text": (choice["message"].get("content") or "").strip(),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": r.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": r.get("usage", {}).get("completion_tokens", 0),
            "timings": r.get("timings", {}),
        }
