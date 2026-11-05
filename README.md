# hf-lambda-mcp

Serve small [Hugging Face](https://huggingface.co) models as **MCP tools on AWS Lambda**, behind
[Amazon Bedrock AgentCore Gateway](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html).

Let your frontier agent (Claude Opus, Sonnet, ...) **delegate bulk, well-defined work to a small,
cheap, serverless model**: translation today, more task-specific models tomorrow. One model = one
YAML *profile*; no model-specific code.

The first profile is **[PLaMo 3 Translate](https://huggingface.co/pfnet/plamo-3-translate-2b-GGUF)**
by Preferred Networks, which translates between 53 languages.

[![Launch Stack](https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png)](https://us-east-2.console.aws.amazon.com/cloudformation/home?region=us-east-2#/stacks/create?stackName=plamo-translate-mcp&templateURL=TEMPLATE_URL)

> [!IMPORTANT]
> **Model licenses.** This repository (Apache-2.0) only contains hosting code. The deployment downloads
> model weights from Hugging Face **into your AWS account**. Each model has its own license, which
> you must review and comply with. For example, PLaMo 3 Translate is distributed under the
> [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en): commercial
> use requires registration with PFN. The stack output `ModelLicense` links the license of the model you deployed.

## How it works

```
                 stdio                 MCP (SigV4)                      invoke
┌────────────┐        ┌──────────────┐           ┌──────────────────┐        ┌────────────────────────────┐
│  AI agent  │◄──────►│ hf-lambda-mcp│◄─────────►│ AgentCore Gateway│───────►│ Lambda (container image)   │
│(Claude Code│        │ local client │           │   (AWS_IAM auth) │        │  llama-server (llama.cpp)  │
│  Desktop..)│        │  (uvx)       │           └──────────────────┘        │  + GGUF model, SnapStart   │
└────────────┘        └──────┬───────┘                                     └─────────────┬──────────────┘
                             │ presigned PUT / GET (files never pass through the agent)  │ async jobs
                             └──────────────────────►  S3 bucket (uploads / results, 1-day expiry) ◄─┘
```

- **One-click deploy.** The CloudFormation template creates a CodeBuild project that resolves the
  profile, downloads the model from Hugging Face, compiles llama.cpp and builds the Lambda image.
  The stack waits for the build and then creates the Lambda function, the Gateway and its target.
- **Lambda SnapStart.** The function is snapshotted *after* the model is loaded, so cold starts restore
  in about 2 seconds instead of reloading gigabytes from the image.
- **Model-agnostic server.** Prompt format, parameter names and input validation come from the model's
  own chat template and GGUF metadata; the profile only adds what those artifacts cannot say
  (agent-facing descriptions, which file to deploy, document handling).
- **Files without tokens.** The local client adds a `<tool>_file` variant for every document-capable
  tool. It uploads a local file through a presigned URL, runs an asynchronous job and writes the result
  next to the input. Only the output path and statistics come back to the agent. Long documents are
  split into contiguous segments that are **processed in parallel** by separate Lambda invocations.

## Quick start

### 1. Deploy

Click **Launch Stack** (or deploy `deployments/template.yaml` yourself) and choose:

| Parameter | Default | Notes |
|---|---|---|
| `ModelProfile` | `plamo-3-translate` | A file in [`profiles/`](profiles) |
| `ModelVariant` | `2b` | `0.8b` (fastest), `2b` (recommended), `8b` (best quality, slow on CPU) |
| `MemorySize` | `10240` | vCPUs scale with memory; 10240 MB = 6 vCPUs, the fastest option |
| `Architecture` | `x86_64` | `arm64` is also supported |
| `EnableSnapStart` | `true` | Keep enabled; see benchmarks |

Deployment takes about 25–35 minutes (image build ≈ 15 min, SnapStart snapshot ≈ 5–10 min).

### 2. Connect your agent

Requirements: [uv](https://docs.astral.sh/uv/) and AWS credentials that can call
`bedrock-agentcore:InvokeGateway` on the gateway (the same credentials you use for the AWS CLI).

Copy the `ClientCommand` stack output. For **Claude Code** (`.mcp.json`):

```json
{
  "mcpServers": {
    "plamo-translate": {
      "command": "uvx",
      "args": [
        "--from", "git+https://github.com/icoxfog417/hf-lambda-mcp@main",
        "hf-lambda-mcp",
        "<McpEndpointUrl stack output>"
      ]
    }
  }
}
```

Options: `--profile <aws-profile>`, `--region` (default: from the URL), `--wait <seconds>` (how long a
`*_file` tool waits before returning a job id), `--expose-job-tools`.

Any MCP client that supports SigV4 can also use the endpoint directly, for example
[mcp-proxy-for-aws](https://github.com/aws/mcp-proxy-for-aws). You then get the remote tools only, without the local `*_file` tools.

### 3. Use it

> Translate docs/guide.md into English with plamo-translate.

The agent calls `translate_file(input_path="docs/guide.md", target_lang="en")` and gets back
`{"output_path": "docs/guide.en.md", "chunks": 12, ...}`.

## Tools (PLaMo 3 Translate profile)

| Tool | Where | Description |
|---|---|---|
| `translate` | Gateway | Translate `text` to `target_lang`; optional `source_lang`, `context`, `glossary`, `form` |
| `translate_file` | local client | Translate a local `.txt` / `.md` file; writes `<stem>.<target_lang><ext>` |
| `download_job_result` | local client | Resume waiting for a long-running file job |
| `get_model_info` | Gateway | Model repository, revision, license and capabilities |
| `create_upload`, `start_job`, `get_job` | Gateway | Generic document job API used by the client (hidden by default) |

Markdown handling: front matter, fenced code blocks, HTML comments and `$$` math blocks are copied
verbatim; everything else is translated paragraph-group by paragraph-group, with the preceding chunks
passed as chat history for consistent terminology.

## Benchmarks

AWS Lambda, us-east-2, x86_64, llama.cpp v0.9.0, GGUF Q4_K_M, translating a 250-character Japanese
paragraph into English (≈105 output tokens):

| Variant | Memory (vCPU) | Generation | Prompt processing | Warm request |
|---|---|---|---|---|
| 0.8B | 10240 MB (6) | 58 tok/s | 117 tok/s | 1.8 s |
| 0.8B | 3538 MB (3) | 25 tok/s | 50–97 tok/s | 4.2 s |
| 2B | 10240 MB (6) | 19 tok/s | 65–69 tok/s | 5.5–7 s |
| 2B | 5308 MB (4) | 11 tok/s | 19–35 tok/s | 9.8 s |

Cold start, 2B, 10240 MB:

| | Without SnapStart | With SnapStart |
|---|---|---|
| New execution environment | **415 s** (model is lazily read from the image) | **2.2 s** restore |
| First request after a cold start (wall clock) | ~420 s | 11 s |

Peak memory for the 2B model is about 3.1 GB. Larger memory settings cost about the same per request,
because the requests finish faster.

## Adding a model

Create `profiles/<name>.yaml`. The profile should contain only what the Hugging Face artifacts cannot provide:

```yaml
name: my-model
source:
  repo: org/model-{variant}-GGUF        # Hugging Face repository with GGUF files
  file: model-{variant}-Q4_K_M.gguf
  variants: [1b, 3b]
  default_variant: 1b
generation: {temperature: 0}            # optional sampling overrides
tools:
  - name: summarize
    description: Agent-facing description ...
    input: {name: text, description: Text to process}
    params:                             # must be variables of the model's chat template
      style: {type: string, description: ...}
    document:                           # optional: enables file jobs and the *_file tool
      formats: [txt, md]
      max_chunk_tokens: 1024
      history: 1
      output_name: "{stem}.summary{ext}"
```

Run `uv run --extra build hf-lambda-mcp-build profiles/my-model.yaml --no-download` to check the profile against
the model's chat template, then deploy with `ModelProfile=my-model`.

Practical limits: Lambda has no GPU and at most 10 GB of memory and 6 vCPUs. Inference speed is roughly
inversely proportional to the model file size; models of about 0.5–3 GB (≈0.5B–4B parameters at 4-bit) work well.

## Development

```bash
uv sync --all-extras                 # client + build + dev dependencies
uv run pytest                        # unit tests (no AWS needed)
uv run cfn-lint deployments/template.yaml

# Build an image locally
uv run --extra build hf-lambda-mcp-build profiles/plamo-3-translate.yaml --variant 0.8b --out build
docker buildx build --platform linux/amd64 --provenance=false -f docker/Dockerfile -t hf-lambda-mcp:local .
docker run --rm -p 9000:8080 hf-lambda-mcp:local
curl -s localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"__tool__": "translate", "text": "こんにちは", "target_lang": "en"}'
```

See [spec/design.md](spec/design.md) for the architecture and the design decisions.

## License

Apache License 2.0. See [LICENSE](LICENSE). Model weights are **not** part of this repository and are
subject to their own licenses.
