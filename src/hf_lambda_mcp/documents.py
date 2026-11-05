"""Model-agnostic document splitting and reassembly.

A document is parsed into a flat list of items. Each item is either *prose* (sent to the
model) or *raw* (copied verbatim: blank-line separators, fenced code blocks, front matter,
HTML comments, math blocks). Consecutive prose items separated only by whitespace are grouped
into chunks that fit ``max_chunk_tokens``; each chunk is processed independently and spliced
back in place, so everything that is not prose round-trips byte-for-byte.

Only the standard library is used so this module runs both in Lambda and on the client.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

SUPPORTED_FORMATS = ("txt", "md")

_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SENTENCE_RE = re.compile(r"(?<=[。！？!?\.])\s*")


@dataclass
class Item:
    text: str
    prose: bool


@dataclass
class Chunk:
    """A contiguous run of items [start, end) whose text is processed by the model."""

    start: int
    end: int
    text: str


@dataclass
class Document:
    items: list[Item]
    chunks: list[Chunk] = field(default_factory=list)

    def assemble(self, outputs: Iterable[str]) -> str:
        """Rebuild the document, replacing each chunk with the corresponding output."""
        outputs = list(outputs)
        if len(outputs) != len(self.chunks):
            raise ValueError(f"expected {len(self.chunks)} outputs, got {len(outputs)}")
        parts: list[str] = []
        pos = 0
        for chunk, out in zip(self.chunks, outputs):
            parts.extend(i.text for i in self.items[pos : chunk.start])
            parts.append(out)
            pos = chunk.end
        parts.extend(i.text for i in self.items[pos:])
        return "".join(parts)


def format_from_name(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return {"markdown": "md", "text": "txt"}.get(ext, ext)


def parse(text: str, fmt: str) -> list[Item]:
    """Split ``text`` into prose/raw items. ``"".join(i.text for i in items) == text`` (after newline normalisation)."""
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"unsupported format '{fmt}'; supported: {', '.join(SUPPORTED_FORMATS)}")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines(keepends=True)
    items: list[Item] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            items.append(Item("".join(para), True))
            para.clear()

    def raw(s: str) -> None:
        if items and not items[-1].prose:
            items[-1].text += s
        else:
            items.append(Item(s, False))

    i = 0
    if fmt == "md" and lines and lines[0].rstrip("\n") == "---":
        # YAML front matter
        for j in range(1, len(lines)):
            if lines[j].rstrip("\n") in ("---", "..."):
                raw("".join(lines[: j + 1]))
                i = j + 1
                break

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            raw(line)
            i += 1
            continue
        if fmt == "md":
            end = _raw_block_end(lines, i)
            if end is not None:
                flush()
                raw("".join(lines[i:end]))
                i = end
                continue
        para.append(line)
        i += 1
    flush()
    return _move_trailing_newline(items)


def _raw_block_end(lines: list[str], i: int) -> int | None:
    """If a Markdown raw block starts at line ``i``, return the index after it."""
    line = lines[i]
    m = _FENCE_RE.match(line)
    if m:
        fence = m.group(1)
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if s.startswith(fence[0] * len(fence)) and not s.strip(fence[0]):
                return j + 1
        return len(lines)
    s = line.strip()
    if s.startswith("<!--"):
        for j in range(i, len(lines)):
            if "-->" in lines[j]:
                return j + 1
        return len(lines)
    if s == "$$":
        for j in range(i + 1, len(lines)):
            if lines[j].strip() == "$$":
                return j + 1
        return len(lines)
    return None


def _move_trailing_newline(items: list[Item]) -> list[Item]:
    """Keep the final newline of a paragraph outside the prose so models don't need to reproduce it."""
    merged: list[Item] = []
    for it in items:
        pieces = [it]
        if it.prose and it.text.endswith("\n"):
            body = it.text.rstrip("\n")
            pieces = [Item(body, True), Item(it.text[len(body) :], False)]
        for p in pieces:
            if merged and not p.prose and not merged[-1].prose:
                merged[-1].text += p.text
            else:
                merged.append(p)
    return merged


def build(text: str, fmt: str, max_tokens: int, count_tokens: Callable[[str], int]) -> Document:
    """Parse and group prose into chunks no larger than ``max_tokens`` (best effort)."""
    items = _split_oversized(parse(text, fmt), max_tokens, count_tokens)
    doc = Document(items)
    start: int | None = None
    tokens = 0
    for idx, it in enumerate(items):
        if it.prose:
            n = count_tokens(it.text)
            if start is not None and tokens + n > max_tokens:
                _close(doc, start, _last_prose(items, start, idx))
                start, tokens = None, 0
            if start is None:
                start = idx
            tokens += n
        elif start is not None and it.text.strip():
            # non-whitespace raw block (code, front matter...) ends the chunk
            _close(doc, start, _last_prose(items, start, idx))
            start, tokens = None, 0
    if start is not None:
        _close(doc, start, _last_prose(items, start, len(items)))
    return doc


def _last_prose(items: list[Item], start: int, stop: int) -> int:
    for j in range(stop - 1, start - 1, -1):
        if items[j].prose:
            return j + 1
    return start + 1


def _close(doc: Document, start: int, end: int) -> None:
    doc.chunks.append(Chunk(start, end, "".join(i.text for i in doc.items[start:end])))


def _split_oversized(items: list[Item], max_tokens: int, count_tokens: Callable[[str], int]) -> list[Item]:
    """Split single prose items larger than ``max_tokens`` at line, then sentence, boundaries."""
    out: list[Item] = []
    for it in items:
        if not it.prose or count_tokens(it.text) <= max_tokens:
            out.append(it)
            continue
        for piece, sep in _pieces(it.text, max_tokens, count_tokens):
            out.append(Item(piece, True))
            if sep:
                out.append(Item(sep, False))
    return out


def _pieces(text: str, max_tokens: int, count_tokens: Callable[[str], int], level: int = 0) -> list[tuple[str, str]]:
    """Return (piece, separator) pairs that concatenate back to ``text``, each piece within budget."""
    splitters = (_split_lines, _split_sentences, _split_hard)
    parts = splitters[level](text)
    if len(parts) == 1 and level < len(splitters) - 1:
        return _pieces(text, max_tokens, count_tokens, level + 1)
    expanded: list[tuple[str, str]] = []
    for piece, sep in parts:
        if level < len(splitters) - 1 and count_tokens(piece) > max_tokens:
            sub = _pieces(piece, max_tokens, count_tokens, level + 1)
            sub[-1] = (sub[-1][0], sub[-1][1] + sep)
            expanded.extend(sub)
        else:
            expanded.append((piece, sep))
    result: list[tuple[str, str]] = []
    buf: str | None = None
    buf_sep = ""
    for piece, sep in expanded:
        if buf is not None and count_tokens(buf + buf_sep + piece) <= max_tokens:
            buf, buf_sep = buf + buf_sep + piece, sep
            continue
        if buf is not None:
            result.append((buf, buf_sep))
        buf, buf_sep = piece, sep
    if buf is not None:
        result.append((buf, buf_sep))
    return result


def _split_lines(text: str) -> list[tuple[str, str]]:
    lines = text.split("\n")
    return [(l, "\n" if k < len(lines) - 1 else "") for k, l in enumerate(lines) if l or k < len(lines) - 1]


def _split_sentences(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _SENTENCE_RE.finditer(text):
        if m.start() == pos or m.start() >= len(text):
            continue
        out.append((text[pos : m.start()], m.group(0)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], ""))
    return out or [(text, "")]


def _split_hard(text: str, size: int = 800) -> list[tuple[str, str]]:
    return [(text[k : k + size], "") for k in range(0, len(text), size)]
