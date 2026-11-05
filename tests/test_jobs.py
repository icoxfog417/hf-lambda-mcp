import collections
import json

import boto3
import pytest
import requests
from moto import mock_aws

from hf_lambda_mcp.server import jobs
from hf_lambda_mcp.server.engine import ToolInputError

BUCKET = "test-bucket"
DOC = "# Title\n\n" + "\n\n".join(f"paragraph {i} with a few words" for i in range(12)) + "\n\n```\nkeep me\n```\n"


class Ctx:
    invoked_function_arn = "arn:aws:lambda:us-east-2:123456789012:function:test-model:live"

    def __init__(self, remaining_ms=900_000):
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self):
        return self.remaining_ms


@pytest.fixture
def aws(monkeypatch):
    with mock_aws():
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
        monkeypatch.setenv("AWS_REGION", "us-east-2")
        monkeypatch.setattr(jobs, "BUCKET", BUCKET)
        monkeypatch.setattr(jobs, "_s3", None)
        boto3.client("s3", region_name="us-east-2").create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": "us-east-2"})
        queue = collections.deque()
        monkeypatch.setattr(jobs, "_invoke_worker", lambda ctx, payload: queue.append(payload))
        yield queue


def upload(engine, name="doc.md", body=DOC):
    up = jobs.create_upload(engine, {"filename": name})
    r = requests.put(up["upload_url"], data=body.encode())
    assert r.status_code == 200
    return up


def drain(engine, queue, ctx_factory=Ctx, max_steps=200):
    steps = 0
    while queue and steps < max_steps:
        jobs.run_worker(engine, queue.popleft(), ctx_factory())
        steps += 1
    return steps


def test_end_to_end_parallel_job(aws, engine, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_PARALLEL", 3)
    up = upload(engine)
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "en"}}, Ctx())
    assert jobs.get_job(engine, job)["status"] == "queued"
    drain(engine, aws)
    st = jobs.get_job(engine, job)
    assert st["status"] == "succeeded", st
    assert st["output_name"] == "doc.en.md"
    assert st["stats"]["segments"] == 3
    body = requests.get(st["download_url"]).text
    assert body == DOC.replace("keep me", "KEEP ME").upper().replace("KEEP ME", "keep me").replace("```\n", "```\n")
    assert "keep me" in body and "PARAGRAPH 11" in body


def test_segment_history_stays_within_segment(aws, engine, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_PARALLEL", 2)
    engine.runtime["context_length"] = 1000
    up = upload(engine)
    jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "en"}}, Ctx())
    drain(engine, aws)
    first_calls = [c for c in engine.server.calls if len(c["messages"]) == 1]
    assert len(first_calls) == 2  # one fresh start per segment


def test_resume_when_time_runs_out(aws, engine, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_PARALLEL", 1)
    up = upload(engine)
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "en"}}, Ctx())
    steps = drain(engine, aws, ctx_factory=lambda: Ctx(remaining_ms=1000))  # always "almost out of time"
    st = jobs.get_job(engine, job)
    assert st["status"] == "succeeded"
    assert steps > 3  # plan + several resumed segment invocations


def test_duplicate_invocations_are_ignored(aws, engine, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_PARALLEL", 2)
    up = upload(engine)
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "en"}}, Ctx())
    plan = aws[0]
    drain(engine, aws)
    calls = len(engine.server.calls)
    assert jobs.run_worker(engine, plan, Ctx()) == {"ignored": True}
    assert jobs.run_worker(engine, {"job_id": job["job_id"], "step": "segment", "segment": 0, "seq": 0}, Ctx()) == {"ignored": True}
    assert len(engine.server.calls) == calls


def test_model_error_marks_job_failed(aws, engine):
    def boom(*a, **k):
        raise RuntimeError("template says no")
    engine.server.chat = boom
    up = upload(engine)
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "en"}}, Ctx())
    drain(engine, aws)
    st = jobs.get_job(engine, job)
    assert st["status"] == "failed" and "template says no" in st["error"]


def test_txt_and_empty_documents(aws, engine):
    up = upload(engine, "empty.txt", "\n\n")
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": {"target_lang": "ja"}}, Ctx())
    drain(engine, aws)
    st = jobs.get_job(engine, job)
    assert st["status"] == "succeeded" and st["output_name"] == "empty.ja.txt"


@pytest.mark.parametrize("args,match", [
    ({"tool": "translate", "key": "other/x.md", "arguments": {"target_lang": "en"}}, "create_upload"),
    ({"tool": "translate", "key": "uploads/abc/missing.md", "arguments": {"target_lang": "en"}}, "not found"),
    ({"tool": "translate", "key": "uploads/abc/x.md", "arguments": {}}, "target_lang"),
    ({"tool": "nope", "key": "uploads/abc/x.md", "arguments": {}}, "unknown tool"),
])
def test_start_job_validation(aws, engine, args, match):
    with pytest.raises(ToolInputError, match=match):
        jobs.start_job(engine, args, Ctx())


def test_create_upload_rejects_unsupported_and_sanitizes(aws, engine):
    with pytest.raises(ToolInputError):
        jobs.create_upload(engine, {"filename": "x.pdf"})
    up = jobs.create_upload(engine, {"filename": "../../etc/pass wd.md"})
    assert up["key"].endswith("/pass_wd.md") and ".." not in up["key"]


def test_arguments_may_be_json_string(aws, engine):
    up = upload(engine)
    job = jobs.start_job(engine, {"tool": "translate", "key": up["key"], "arguments": json.dumps({"target_lang": "en"})}, Ctx())
    assert job["status"] == "queued"


def test_plan_segments():
    assert jobs.plan_segments(0) == []
    assert jobs.plan_segments(1) == [[0, 1]]
    assert jobs.plan_segments(5, 8, 2) == [[0, 3], [3, 5]]
    segs = jobs.plan_segments(100, 8, 2)
    assert len(segs) == 8 and segs[0][0] == 0 and segs[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))


def test_output_name():
    assert jobs.output_name("{stem}.{target_lang}{ext}", "uploads/x/report.md", {"target_lang": "zh-hans"}) == "report.zh-hans.md"
    assert jobs.output_name("{stem}.{missing}{ext}", "uploads/x/a.txt", {}) == "a..txt"
