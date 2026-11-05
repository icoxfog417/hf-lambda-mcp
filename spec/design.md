# Design

## Goal

Let a frontier agent delegate bulk, well-defined work, translation first, to a small open model running
on serverless infrastructure, with one-click deployment and no per-model code. PLaMo 3 Translate is the
first model; the architecture is a general "Hugging Face model → MCP tool" path.

## Components

| Component | Responsibility |
|---|---|
| `profiles/<name>.yaml` | What cannot be derived from the model artifacts: which file to deploy, agent-facing tool descriptions and parameter types, document handling |
| `hf-lambda-mcp-build` (`profile.py`) | Resolve the profile against Hugging Face, download the GGUF and chat template, check the parameters against the template variables, emit `runtime.json` and the Gateway tool schema |
| Lambda image (`docker/Dockerfile`) | llama.cpp `llama-server` (all CPU variants, chosen at runtime) + model + `hf_lambda_mcp.server` |
| Lambda handler (`server/`) | Gateway tool calls; inline execution; asynchronous S3-backed document jobs |
| AgentCore Gateway | MCP endpoint, AWS_IAM inbound auth, Lambda target with S3-hosted tool schema |
| Local client (`client/cli.py`) | stdio MCP server: proxies remote tools with SigV4 and adds `<tool>_file` tools that move files through presigned URLs |
| `deployments/HFLambdaMCP.yaml` | One stack: S3, ECR, CodeBuild, build coordinator, Lambda (+version/alias), Gateway, target |

## Key decisions

### What a profile contains (and what it doesn't)

The model card is prose; the artifacts are machine-readable. Derived automatically:

- Prompt format: `chat_template.jinja` (or `tokenizer.chat_template` in the GGUF), passed to llama-server with `--jinja`.
- Parameter names: undeclared variables of the template (variables assigned inside the template are excluded,
  unless they are tested with `is defined`). The build fails if a profile parameter is not a template variable.
- Validation: the template's own `raise_exception` calls. llama-server returns these as HTTP 400, and the
  handler returns them as tool errors, so the agent can correct its call.
- Context length: GGUF metadata. Runtime default: `min(model, 8192)`.
- License: Hugging Face card metadata (`license`, `license_name`, `license_link`).
- Response: the assistant message text. No configuration needed. PLaMo stops correctly on its EOS token
  without the extra stop strings shown in its model card.

### Container image in ECR, not EFS

- Lambda memory (10 GB) is the binding limit, not the 10 GB image limit: the model plus KV cache must
  fit in RAM, so a model too large for the image would not run anyway.
- Non-model overhead is small: llama.cpp with all CPU variants is 71 MB; the Python base is about 0.6 GB.
- CPU decoding speed is roughly inversely proportional to model size; usable models are ≤ ~3–5 GB.
- EFS would add a VPC, a provisioner step and slower network reads on every cold start.

### Cold start: prefetch by default, SnapStart opt-in

Measured (2B Q4_K_M, 10240 MB, x86_64):

| Strategy | Cold start (wall) | Idle cost |
|---|---|---|
| Plain (llama.cpp mmap against a lazily loaded image) | ~415 s | none |
| **Parallel sequential prefetch** of the GGUF before `llama-server` starts (default) | **~5–80 s** (depends on Lambda's image cache) | none |
| SnapStart (snapshot after load + warm-up) | ~11 s (2.2 s restore) | snapshot cache ≈ $0.0000015/GB-s ≈ **$40/month at 10 GB** |

`llama-server` runs as a child process and survives a SnapStart snapshot and restore. The template
parameter `EnableSnapStart` lets users trade idle cost for latency.

The very first start of a newly pushed image is slow (≈170 s for 0.8B, ≈420 s for 2B), because Lambda pulls it
from ECR for the first time. The `WarmUp` custom resource invokes each new version once before the Gateway target is
created/updated. This absorbs that delay, and it fails the deployment if the model cannot load. Prefetch thread count
(8–64) made no consistent difference; caching on the Lambda host dominates.

### Throughput and parallel documents

Warm, 6 vCPUs, x86_64: 2B ≈ 19 tok/s generation and ≈ 69 tok/s prompt processing; 0.8B ≈ 58 and ≈ 117 tok/s.
Thread count, flash attention and batch size made no measurable difference on x86 (no AVX-512); the CPU is the limit.
arm64 (Graviton) was faster: a 2B paragraph took 2.7 s against 6.5 s on x86, with prompt processing at 104 tok/s.
arm64 is also ~20% cheaper per GB-second, so it is the default. The arm64 image uses a single
`armv8.2-a+dotprod+fp16` build, because AL2023's GCC 11 cannot compile llama.cpp's armv9.2 variant.

Consequences:

- Document jobs split the chunk list into up to `MAX_PARALLEL` (template parameter `MaxParallelSegments`,
  default 8) contiguous segments, each processed by its own asynchronous Lambda invocation. The total compute is the same;
  the wall-clock time is divided by the number of segments. Measured: an 18 KB Markdown file as 3 sequential 1024-token
  chunks took 305 s with 2B on x86, so the PLaMo profile uses 512-token chunks with one segment per chunk.
- Within a segment, previous (input, output) pairs are sent as chat history for consistent terminology.
  The window **grows** (rather than sliding) while it fits in half of the context. Consecutive requests then share
  a prefix, and llama-server reuses its KV cache instead of re-processing ~2k prompt tokens per chunk.
  When the budget is exceeded, the window restarts from the last `history` pairs.
- Each segment worker checkpoints after every chunk to S3 and re-invokes itself before the 15-minute
  Lambda timeout, so segments of any size complete. Duplicate invocations are ignored (sequence numbers).
  Async retries are disabled; a worker that stops updating for 16 minutes is reported as failed.
- The worker that finishes the last segment assembles the result. Two workers racing produce the same output, so this is safe.

### Document format handling

`documents.py` splits into prose and raw items. Raw items are copied byte for byte: blank lines, YAML front
matter, fenced code blocks, HTML comments, `$$` math blocks. Prose paragraphs are grouped up to
`max_chunk_tokens` (counted with llama-server's own tokenizer). Paragraphs that are too long are split at line, then sentence,
then fixed-size boundaries. Reassembly is lossless when outputs equal inputs (tested).

### Files never pass through the agent

Passing a long document as a tool argument costs the agent as many output tokens as translating it itself
would. The local client therefore reads the file, uploads it with a presigned PUT to the stack's bucket
(users need no S3 permissions or bucket of their own), starts a job, polls (sending MCP progress
notifications), and writes the result next to the input. Only the path and statistics reach the agent.
`*_file` tools are generated from `get_model_info`, so the client contains no model-specific logic.

### Authentication

AWS_IAM (SigV4) inbound auth on the Gateway. OAuth via Cognito was rejected because it lacks
DCR/CIMD, which MCP clients such as Claude require. The local client reuses `mcp-proxy-for-aws`'s
`aws_iam_streamablehttp_client`.

### One-click deployment flow

1. The `ModelImage` custom resource starts CodeBuild and passes the pending CloudFormation request in an environment variable.
2. CodeBuild clones the repository at `SourceRef`, runs `hf-lambda-mcp-build`, builds and pushes the image, uploads
   the tool schema to S3 and writes `build/results/<build-id>.json`.
3. An EventBridge rule for CodeBuild state changes invokes the coordinator, which answers CloudFormation:
   SUCCESS with the image URI, schema URI and license link, or FAILED with a link to the logs.
   This covers failures in any build phase.
4. CloudFormation creates the function, a version (a new one whenever the image tag changes), the `live` alias,
   the Gateway (AWS_IAM) and the Lambda target, which uses the S3 tool schema.

Model weights are downloaded inside the user's account at deploy time and never redistributed by this project.

## Limits and future work

- CPU only. For models above ~4B parameters, use a GPU backend (e.g. AgentCore Runtime or SageMaker) behind the same Gateway.
- Formats: `.txt`, `.md`. Next: `.docx`, `.html` (a new splitter in `documents.py`; profiles unchanged).
- `hf-lambda-mcp init <repo>`: draft a profile from the chat template and GGUF metadata.
- Several profiles behind one Gateway (one target each).
