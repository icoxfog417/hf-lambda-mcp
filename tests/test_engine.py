import pytest

from hf_lambda_mcp.server.engine import ToolInputError

TEXT = "\n\n".join(f"para {i} has some words" for i in range(6))


def test_inline_translate_chunks_and_reassembles(engine):
    r = engine.run_inline("translate", {"text": TEXT, "target_lang": "en"})
    assert r["output"] == TEXT.upper()
    assert r["chunks"] == 3
    assert all(c["kwargs"] == {"target_lang": "en"} for c in engine.server.calls)


def test_history_is_passed_and_grows_for_cache_reuse(engine):
    engine.runtime["context_length"] = 400
    engine.run_inline("translate", {"text": TEXT, "target_lang": "en"})
    lens = [len(c["messages"]) for c in engine.server.calls]
    assert lens[0] == 1
    # history: previous pairs as user/assistant turns, kept growing while within budget
    assert lens[1] == 3 and lens[2] == 5
    assert engine.server.calls[1]["messages"][1]["role"] == "assistant"


def test_history_window_resets_when_over_budget(engine):
    engine.runtime["context_length"] = 30  # budget 15 tokens -> window restarts
    text = "\n\n".join(f"p{i} " + "w " * 10 for i in range(6))
    engine.tools["translate"]["document"]["max_chunk_tokens"] = 11
    engine.run_inline("translate", {"text": text, "target_lang": "en"})
    assert max(len(c["messages"]) for c in engine.server.calls) == 3  # never more than history=1 pair


def test_optional_empty_args_dropped(engine):
    engine.run_inline("translate", {"text": "hello", "target_lang": "ja", "source_lang": "", "glossary": [], "form": None})
    assert engine.server.calls[0]["kwargs"] == {"target_lang": "ja"}


@pytest.mark.parametrize("args,match", [
    ({"text": "hi"}, "target_lang"),
    ({"text": "", "target_lang": "en"}, "non-empty"),
    ({"text": "hi", "target_lang": "en", "bogus": 1}, "bogus"),
])
def test_input_validation(engine, args, match):
    with pytest.raises(ToolInputError, match=match):
        engine.run_inline("translate", args)


def test_unknown_tool(engine):
    with pytest.raises(ToolInputError):
        engine.run_inline("summarize", {"text": "x"})


def test_code_blocks_are_not_sent_to_model(engine):
    md = "Intro text\n\n```\ncode here\n```\n\nOutro text\n"
    r = engine.run_inline("translate", {"text": md, "target_lang": "en"})
    assert r["output"] == "INTRO TEXT\n\n```\ncode here\n```\n\nOUTRO TEXT\n"
    assert not any("code here" in m["content"] for c in engine.server.calls for m in c["messages"])
