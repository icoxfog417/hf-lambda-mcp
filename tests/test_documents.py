import pytest

from hf_lambda_mcp import documents as d

MD = """---
title: Test
tags: [a, b]
---

# Heading

First paragraph. Second sentence.
Continued line.

- item 1
- item 2

```python
print("code is never translated")
```

<!-- comment -->
$$
E = mc^2
$$

Last paragraph.
"""


def chars(s: str) -> int:
    return len(s)


def test_markdown_roundtrip_and_raw_blocks():
    doc = d.build(MD, "md", 1000, chars)
    assert doc.assemble(c.text for c in doc.chunks) == MD
    joined = "".join(c.text for c in doc.chunks)
    assert "print(" not in joined and "title:" not in joined and "E = mc^2" not in joined and "comment" not in joined
    assert "# Heading" in joined and "Last paragraph." in joined


def test_markdown_replacement_keeps_structure():
    doc = d.build(MD, "md", 1000, chars)
    out = doc.assemble(f"<{i}>" for i in range(len(doc.chunks)))
    assert out.startswith("---\ntitle: Test")
    assert '```python\nprint("code is never translated")\n```' in out
    assert out.endswith("<%d>\n" % (len(doc.chunks) - 1))


def test_chunks_respect_budget_and_group_paragraphs():
    text = "\n\n".join(f"Paragraph {i} " + "x" * 40 for i in range(10))
    doc = d.build(text, "txt", 120, chars)
    assert len(doc.chunks) > 1
    assert all(chars(c.text) <= 120 for c in doc.chunks)
    assert doc.assemble(c.text for c in doc.chunks) == text
    # paragraphs are grouped: more than one paragraph per chunk
    assert "\n\n" in doc.chunks[0].text


@pytest.mark.parametrize("text", [
    "あ" * 50 + "。" + "い" * 30 + "。\n" + "う" * 90,
    "line one\nline two\n" * 30,
    "word " * 500,
])
def test_oversized_paragraph_is_split_losslessly(text):
    doc = d.build(text, "txt", 60, chars)
    assert doc.assemble(c.text for c in doc.chunks) == text
    assert len(doc.chunks) > 1


def test_crlf_normalized_and_empty():
    assert d.build("a\r\n\r\nb", "txt", 100, chars).assemble(["A\n\nB"]) == "A\n\nB"
    assert d.build("", "md", 100, chars).chunks == []
    assert d.build("\n\n", "md", 100, chars).chunks == []


def test_unclosed_fence_is_raw():
    text = "intro\n\n```\nunterminated code"
    doc = d.build(text, "md", 100, chars)
    assert [c.text for c in doc.chunks] == ["intro"]


def test_unsupported_format():
    with pytest.raises(ValueError):
        d.parse("x", "pdf")


def test_format_from_name():
    assert d.format_from_name("a/b/Report.MD") == "md"
    assert d.format_from_name("notes.markdown") == "md"
    assert d.format_from_name("x.txt") == "txt"
