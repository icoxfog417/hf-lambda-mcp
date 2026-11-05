"""Generic asynchronous document jobs backed by S3.

Layout in the stack bucket (all prefixes expire via lifecycle rules):

    uploads/<id>/<filename>          input written by the client through a presigned PUT
    jobs/<job_id>.json               job record (status, segments, arguments)
    jobs/<job_id>/seg-<n>.json       per-segment outputs and progress (written by one worker only)
    results/<job_id>/<output_name>   final output, downloaded through a presigned GET

Workers are asynchronous self-invocations of the same Lambda function:

1. *plan*: split the document into chunks, partition the chunks into up to ``MAX_PARALLEL``
   contiguous segments, and start one worker per segment.
2. *segment*: process its chunks sequentially (chat history stays within the segment). When the
   remaining time is short it saves its state and re-invokes itself, so segments of any length fit
   Lambda's 15-minute limit.
3. The worker that completes the last segment assembles the result (idempotent if two race).
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .. import documents
from .engine import Engine, RunStats, ToolInputError

BUCKET = os.environ.get("JOB_BUCKET", "")
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
UPLOAD_URL_TTL = 900
DOWNLOAD_URL_TTL = 3600
STALE_AFTER_S = 16 * 60  # a running job not updated for longer than a Lambda timeout is dead
RESUME_MARGIN_MS = 120_000
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "8"))
MIN_CHUNKS_PER_SEGMENT = int(os.environ.get("MIN_CHUNKS_PER_SEGMENT", "2"))

_s3 = None
_lambda = None


def s3():
    global _s3
    if _s3 is None:
        region = os.environ.get("AWS_REGION")
        _s3 = boto3.client("s3", region_name=region, config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    return _s3


def lambda_client():
    global _lambda
    if _lambda is None:
        _lambda = boto3.client("lambda")
    return _lambda


def _require_bucket() -> None:
    if not BUCKET:
        raise ToolInputError("document jobs are not configured (JOB_BUCKET is not set)")


def _put_json(key: str, obj: Any) -> None:
    s3().put_object(Bucket=BUCKET, Key=key, Body=json.dumps(obj, ensure_ascii=False).encode(), ContentType="application/json")


def _get_json(key: str) -> Any:
    return json.loads(s3().get_object(Bucket=BUCKET, Key=key)["Body"].read())


_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    name = _SAFE.sub("_", name).strip("._") or "document"
    return name[:120]


# ---------------------------------------------------------------------------- tools


def create_upload(engine: Engine, args: dict[str, Any]) -> dict[str, Any]:
    _require_bucket()
    filename = _safe_name(str(args.get("filename") or ""))
    fmt = documents.format_from_name(filename)
    supported = sorted({f for t in engine.tools.values() for f in (t.get("document") or {}).get("formats", [])})
    if fmt not in supported:
        raise ToolInputError(f"unsupported file type '{filename}'; supported extensions: {supported}")
    key = f"uploads/{uuid.uuid4().hex}/{filename}"
    url = s3().generate_presigned_url(
        "put_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=UPLOAD_URL_TTL
    )
    return {"upload_url": url, "method": "PUT", "key": key, "expires_in": UPLOAD_URL_TTL, "max_bytes": MAX_UPLOAD_BYTES}


def start_job(engine: Engine, args: dict[str, Any], context: Any) -> dict[str, Any]:
    _require_bucket()
    tool = engine.tool(str(args.get("tool") or ""))
    doc_cfg = tool.get("document")
    if not doc_cfg:
        raise ToolInputError(f"tool '{tool['name']}' does not support documents")
    key = str(args.get("key") or "")
    if not key.startswith("uploads/"):
        raise ToolInputError("key must be the value returned by create_upload")
    fmt = documents.format_from_name(key)
    if fmt not in doc_cfg.get("formats", []):
        raise ToolInputError(f"tool '{tool['name']}' supports {doc_cfg.get('formats')}, not '{fmt}'")
    arguments = args.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            raise ToolInputError("arguments must be an object") from None
    arguments.pop(tool["input"]["name"], None)
    engine.template_kwargs(tool, arguments)  # validate early
    try:
        head = s3().head_object(Bucket=BUCKET, Key=key)
    except ClientError:
        raise ToolInputError(f"uploaded object not found: {key} (did the PUT succeed?)") from None
    if head["ContentLength"] > MAX_UPLOAD_BYTES:
        raise ToolInputError(f"file is too large ({head['ContentLength']} bytes > {MAX_UPLOAD_BYTES})")

    job_id = uuid.uuid4().hex
    now = time.time()
    job = {
        "job_id": job_id, "status": "queued", "tool": tool["name"], "key": key, "format": fmt,
        "arguments": arguments, "created_at": now, "updated_at": now, "segments": None,
        "output_template": doc_cfg.get("output_name", "{stem}.out{ext}"),
    }
    _put_json(f"jobs/{job_id}.json", job)
    _invoke_worker(context, {"job_id": job_id, "step": "plan"})
    return {"job_id": job_id, "status": "queued"}


def get_job(engine: Engine, args: dict[str, Any]) -> dict[str, Any]:
    _require_bucket()
    job_id = str(args.get("job_id") or "")
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise ToolInputError("invalid job_id")
    try:
        job = _get_json(f"jobs/{job_id}.json")
    except ClientError:
        raise ToolInputError(f"job not found: {job_id}") from None
    progress = {"done": 0, "total": None}
    updated = job["updated_at"]
    if job["status"] == "running" and job.get("segments"):
        progress["total"] = sum(e - st for st, e in job["segments"])
        for n in range(len(job["segments"])):
            seg = _get_segment(job_id, n)
            if seg:
                progress["done"] += len(seg["outputs"])
                updated = max(updated, seg["updated_at"])
                if seg.get("error"):
                    job.update(status="failed", error=seg["error"])
    elif job["status"] == "succeeded":
        progress = {"done": job["stats"]["chunks"], "total": job["stats"]["chunks"]}
    job["progress"] = progress
    if job["status"] in ("queued", "running") and time.time() - updated > STALE_AFTER_S:
        job["status"] = "failed"
        job["error"] = "job stopped responding (worker timed out or crashed)"
    out = {k: job.get(k) for k in ("job_id", "status", "tool", "progress", "error", "output_name", "stats")}
    if job["status"] == "succeeded":
        out["download_url"] = s3().generate_presigned_url(
            "get_object",
            Params={
                "Bucket": BUCKET, "Key": job["result_key"],
                "ResponseContentType": "text/plain; charset=utf-8" if job["format"] == "txt" else "text/markdown; charset=utf-8",
            },
            ExpiresIn=DOWNLOAD_URL_TTL,
        )
    return {k: v for k, v in out.items() if v is not None}


# ---------------------------------------------------------------------------- worker


def _invoke_worker(context: Any, payload: dict[str, Any]) -> None:
    function = getattr(context, "invoked_function_arn", None) or os.environ["AWS_LAMBDA_FUNCTION_NAME"]
    lambda_client().invoke(
        FunctionName=function, InvocationType="Event", Payload=json.dumps({"hf_lambda_mcp_job": payload}).encode()
    )


def _seg_key(job_id: str, n: int) -> str:
    return f"jobs/{job_id}/seg-{n}.json"


def _get_segment(job_id: str, n: int) -> dict[str, Any] | None:
    try:
        return _get_json(_seg_key(job_id, n))
    except ClientError:
        return None


def plan_segments(n_chunks: int, max_parallel: int | None = None, min_chunks: int | None = None) -> list[list[int]]:
    """Partition ``n_chunks`` into contiguous, near-equal [start, end) ranges."""
    max_parallel = MAX_PARALLEL if max_parallel is None else max_parallel
    min_chunks = MIN_CHUNKS_PER_SEGMENT if min_chunks is None else min_chunks
    if n_chunks == 0:
        return []
    k = max(1, min(max_parallel, n_chunks // max(1, min_chunks)))
    base, extra = divmod(n_chunks, k)
    out, pos = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append([pos, pos + size])
        pos += size
    return out


def output_name(template: str, source_key: str, arguments: dict[str, Any]) -> str:
    p = PurePosixPath(source_key)
    values = defaultdict(str, {k: v for k, v in arguments.items() if isinstance(v, (str, int, float))})
    values.update(stem=p.stem, ext=p.suffix, name=p.name)
    return _safe_name(template.format_map(values)) or p.name


def run_worker(engine: Engine, payload: dict[str, Any], context: Any) -> dict[str, Any]:
    job_id = payload["job_id"]
    job_key = f"jobs/{job_id}.json"
    job = _get_json(job_key)
    if job["status"] not in ("queued", "running"):
        return {"ignored": True}
    try:
        tool = engine.tool(job["tool"])
        text = s3().get_object(Bucket=BUCKET, Key=job["key"])["Body"].read().decode("utf-8-sig")
        kwargs = engine.template_kwargs(tool, job["arguments"])
        doc = engine.build_document(tool, text, job["format"])
        if payload.get("step") == "plan":
            return _plan(job, job_key, doc, context)
        return _run_segment(engine, job, job_key, tool, doc, kwargs, int(payload["segment"]), int(payload.get("seq", 0)), context)
    except Exception as e:  # noqa: BLE001 - report every failure on the job record
        job.update(status="failed", error=f"{type(e).__name__}: {e}", updated_at=time.time())
        _put_json(job_key, job)
        return {"failed": True, "error": str(e)}


def _plan(job: dict[str, Any], job_key: str, doc: documents.Document, context: Any) -> dict[str, Any]:
    if job.get("segments") is not None:
        return {"ignored": True}  # duplicate plan invocation
    segments = plan_segments(len(doc.chunks))
    job.update(status="running", segments=segments, updated_at=time.time())
    _put_json(job_key, job)
    if not segments:
        return _assemble(job, job_key, doc, [])
    for n in range(len(segments)):
        _invoke_worker(context, {"job_id": job["job_id"], "step": "segment", "segment": n, "seq": 0})
    return {"planned": len(segments), "chunks": len(doc.chunks)}


def _run_segment(engine: Engine, job: dict[str, Any], job_key: str, tool: dict[str, Any], doc: documents.Document,
                 kwargs: dict[str, Any], n: int, seq: int, context: Any) -> dict[str, Any]:
    job_id = job["job_id"]
    start, end = job["segments"][n]
    seg = _get_segment(job_id, n) or {"outputs": [], "stats": {}, "seq": 0, "done": False}
    if seg.get("seq", 0) != seq or seg.get("done"):
        return {"ignored": True}  # duplicate or superseded invocation
    stats = RunStats(**{k: v for k, v in seg["stats"].items() if k != "started"})
    chunk_times: list[float] = []
    last = [time.time()]

    def save(outs: list[str], st: RunStats, **extra: Any) -> None:
        seg.update(outputs=outs, stats={k: v for k, v in st.__dict__.items() if k != "started"},
                   updated_at=time.time(), **extra)
        _put_json(_seg_key(job_id, n), seg)

    def on_chunk(idx: int, outs: list[str], st: RunStats) -> bool:
        now = time.time()
        chunk_times.append(now - last[0])
        last[0] = now
        save(outs, st)
        if start + len(outs) < end and context is not None:
            if context.get_remaining_time_in_millis() < RESUME_MARGIN_MS + 2000 * max(chunk_times):
                return False
        return True

    save(seg["outputs"], stats)  # heartbeat
    try:
        outputs, stats = engine.run_chunks(tool, doc, kwargs, start, end, seg["outputs"], stats, on_chunk)
    except Exception as e:  # noqa: BLE001
        save(seg["outputs"], stats, error=f"{type(e).__name__}: {e}")
        raise
    if start + len(outputs) < end:
        save(outputs, stats, seq=seq + 1)
        _invoke_worker(context, {"job_id": job_id, "step": "segment", "segment": n, "seq": seq + 1})
        return {"paused": True, "segment": n, "done": len(outputs)}
    save(outputs, stats, done=True)

    # Assemble when every segment is done (whichever worker finishes last).
    all_outputs: list[str] = []
    totals = RunStats(started=job["created_at"])
    for m in range(len(job["segments"])):
        other = outputs if m == n else (_get_segment(job_id, m) or {})
        if m != n and not other.get("done"):
            return {"segment_done": n}
        all_outputs += outputs if m == n else other["outputs"]
        st = stats.__dict__ if m == n else other["stats"]
        totals.chunks += st["chunks"]
        totals.prompt_tokens += st["prompt_tokens"]
        totals.completion_tokens += st["completion_tokens"]
        totals.truncated_chunks += st.get("truncated_chunks", [])
    return _assemble(job, job_key, doc, all_outputs, totals)


def _assemble(job: dict[str, Any], job_key: str, doc: documents.Document, outputs: list[str],
              stats: RunStats | None = None) -> dict[str, Any]:
    stats = stats or RunStats(started=job["created_at"])
    result = doc.assemble(outputs)
    name = output_name(job.get("output_template") or "{stem}.out{ext}", job["key"], job["arguments"])
    result_key = f"results/{job['job_id']}/{name}"
    s3().put_object(Bucket=BUCKET, Key=result_key, Body=result.encode("utf-8"), ContentType="text/plain; charset=utf-8")
    d = stats.as_dict()
    d["segments"] = len(job.get("segments") or [])
    job.update(status="succeeded", result_key=result_key, output_name=name, stats=d, updated_at=time.time())
    _put_json(job_key, job)
    return {"succeeded": True}
