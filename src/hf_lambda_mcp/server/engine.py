"""Model-agnostic tool execution driven by the resolved profile (runtime.json)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .. import documents
from .llama import LlamaServer

MIN_OUTPUT_TOKENS = 256
OUTPUT_RATIO = 4  # safety cap against runaway generation: output <= 4x input tokens per chunk


class ToolInputError(ValueError):
    pass


@dataclass
class RunStats:
    chunks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated_chunks: list[int] = field(default_factory=list)
    started: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "chunks": self.chunks,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(time.time() - self.started, 2),
        }
        if self.truncated_chunks:
            d["warning"] = f"output hit the token limit in chunk(s) {self.truncated_chunks}; result may be incomplete"
        return d


class Engine:
    def __init__(self, runtime: dict[str, Any], server: LlamaServer):
        self.runtime = runtime
        self.server = server
        self.tools = {t["name"]: t for t in runtime["tools"]}

    # ------------------------------------------------------------------ arguments

    def tool(self, name: str) -> dict[str, Any]:
        if name not in self.tools:
            raise ToolInputError(f"unknown tool '{name}'; available: {sorted(self.tools)}")
        return self.tools[name]

    def template_kwargs(self, tool: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
        params = tool.get("params", {})
        unknown = set(args) - set(params) - {tool["input"]["name"]}
        if unknown:
            raise ToolInputError(f"unknown argument(s) {sorted(unknown)} for tool '{tool['name']}'")
        missing = [k for k, spec in params.items() if spec.get("required") and args.get(k) in (None, "")]
        if missing:
            raise ToolInputError(f"missing required argument(s) {missing} for tool '{tool['name']}'")
        return {k: v for k, v in args.items() if k in params and v not in (None, "", [])}

    # ------------------------------------------------------------------ execution

    def build_document(self, tool: dict[str, Any], text: str, fmt: str) -> documents.Document:
        doc_cfg = tool.get("document") or {}
        max_tokens = int(doc_cfg.get("max_chunk_tokens") or self.runtime["context_length"] // 4)
        return documents.build(text, fmt, max_tokens, self.server.count_tokens)

    def run_chunks(
        self,
        tool: dict[str, Any],
        doc: documents.Document,
        kwargs: dict[str, Any],
        start: int = 0,
        end: int | None = None,
        outputs: list[str] | None = None,
        stats: RunStats | None = None,
        on_chunk: Callable[[int, list[str], RunStats], bool] | None = None,
    ) -> tuple[list[str], RunStats]:
        """Process chunks ``[start, end)`` sequentially; ``outputs`` holds results already done for this range.

        History: the previous (input, output) pairs of this range are sent as chat history. At least
        ``document.history`` pairs are kept; more are kept while they fit in half of the context so that
        consecutive requests share a prefix and llama-server can reuse its KV cache. When the budget is
        exceeded, the window restarts from the last ``history`` pairs. ``on_chunk`` returning False pauses.
        """
        end = len(doc.chunks) if end is None else end
        outputs = list(outputs or [])
        stats = stats or RunStats()
        history_n = int((tool.get("document") or {}).get("history", 0))
        budget = self.runtime["context_length"] // 2
        window = start
        pair_tokens: dict[int, int] = {}
        for idx in range(start + len(outputs), end):
            chunk = doc.chunks[idx].text
            n_in = self.server.count_tokens(chunk)
            if history_n <= 0:
                window = idx
            else:
                if idx - window > history_n:
                    used = sum(pair_tokens.get(j) or self._pair_tokens(doc, outputs, start, j, pair_tokens) for j in range(window, idx))
                    if used + n_in > budget:
                        window = idx - history_n
            messages: list[dict[str, str]] = []
            for j in range(window, idx):
                messages += [{"role": "user", "content": doc.chunks[j].text}, {"role": "assistant", "content": outputs[j - start]}]
            messages.append({"role": "user", "content": chunk})
            r = self.server.chat(messages, kwargs, max_tokens=max(MIN_OUTPUT_TOKENS, OUTPUT_RATIO * n_in))
            outputs.append(r["text"])
            pair_tokens[idx] = n_in + r["completion_tokens"]
            stats.chunks += 1
            stats.prompt_tokens += r["prompt_tokens"]
            stats.completion_tokens += r["completion_tokens"]
            if r["finish_reason"] == "length":
                stats.truncated_chunks.append(idx)
            if on_chunk and on_chunk(idx, outputs, stats) is False:
                break
        return outputs, stats

    def _pair_tokens(self, doc: documents.Document, outputs: list[str], start: int, j: int, cache: dict[int, int]) -> int:
        cache[j] = self.server.count_tokens(doc.chunks[j].text) + self.server.count_tokens(outputs[j - start])
        return cache[j]

    def run_inline(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.tool(name)
        input_name = tool["input"]["name"]
        text = args.get(input_name)
        if not isinstance(text, str) or not text.strip():
            raise ToolInputError(f"'{input_name}' must be a non-empty string")
        kwargs = self.template_kwargs(tool, args)
        formats = (tool.get("document") or {}).get("formats", [])
        fmt = "md" if "md" in formats else "txt"
        doc = self.build_document(tool, text, fmt)
        if not doc.chunks:
            return {"output": text, **RunStats().as_dict()}
        outputs, stats = self.run_chunks(tool, doc, kwargs)
        return {"output": doc.assemble(outputs), **stats.as_dict()}

    def warmup(self) -> None:
        """Touch all weights once (so they are resident before a SnapStart snapshot)."""
        self.server._request("POST", "/completion", {"prompt": "Hello", "n_predict": 4, "cache_prompt": False})
