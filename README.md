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
│  Desktop..)│        │  (uvx)       │           └──────────────────┘        │  + GGUF model (prefetched) │
└────────────┘        └──────┬───────┘                                     └─────────────┬──────────────┘
                             │ presigned PUT / GET (files never pass through the agent)  │ async jobs
                             └──────────────────────►  S3 bucket (uploads / results, 1-day expiry) ◄─┘
```

- **One-click deploy.** The CloudFormation template creates a CodeBuild project that resolves the
  profile, downloads the model from Hugging Face, compiles llama.cpp and builds the Lambda image.
  The stack waits for the build and then creates the Lambda function, the Gateway and its target.
- **Fast cold starts, zero idle cost.** The model file is prefetched with parallel sequential reads
  before llama.cpp loads it: a 2B model cold-starts in about 30 s instead of 7 minutes. Optionally,
  Lambda SnapStart restores a snapshot taken after the model has loaded (~11 s), at the price of a snapshot-cache fee while idle.
- **Model-agnostic server.** Prompt format, parameter names and input validation come from the model's
  own chat template and GGUF metadata; the profile only adds what those artifacts cannot say
  (agent-facing descriptions, which file to deploy, document handling).
- **Files without tokens.** The local client adds a `<tool>_file` variant for every document-capable
  tool. It uploads a local file through a presigned URL, runs an asynchronous job and writes the result
  next to the input. Only the output path and statistics come back to the agent. Long documents are
  split into contiguous segments that are **processed in parallel** by separate Lambda invocations.

## Quick start

### 1. Deploy

Click **Launch Stack** (or deploy `deployments/HFLambdaMCP.yaml` yourself) and choose:

| Parameter | Default | Notes |
|---|---|---|
| `ModelProfile` | `plamo-3-translate` | A file in [`profiles/`](profiles) |
| `ModelVariant` | `2b` | `0.8b` (fastest), `2b` (recommended), `8b` (best quality, slow on CPU) |
| `MemorySize` | `10240` | vCPUs scale with memory; 10240 MB = 6 vCPUs, the fastest option |
| `Architecture` | `arm64` | arm64 was 1.5–2.4× faster than `x86_64` in our tests, and is cheaper |
| `MaxParallelSegments` | `8` | Parallel Lambda invocations per document job; `1` = sequential with full chat history |
| `EnableSnapStart` | `false` | `true` = faster cold starts, but an idle cost of ≈ USD 40/month at 10240 MB (see benchmarks) |

Deployment takes about 20–30 minutes (the image build takes most of it; add 5–10 minutes with SnapStart).

### 2. Connect your agent

Requirements: [uv](https://docs.astral.sh/uv/) and AWS credentials (the ones your AWS CLI uses) that are
allowed to call the gateway. Administrators already are; for other identities, attach:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "bedrock-agentcore:InvokeGateway",
    "Resource": "<GatewayArn stack output>"
  }]
}
```

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
verbatim. Everything else is translated in groups of paragraphs of about 512 tokens. Within each parallel segment,
the preceding chunks are passed as chat history for consistent terminology. Use `glossary` to keep key
terms consistent across segments.

## Benchmarks

AWS Lambda in us-east-2, 10240 MB (6 vCPUs), llama.cpp v0.9.0, GGUF Q4_K_M. Task: translate a
250-character Japanese paragraph into English (≈105 output tokens). Times are for warm requests.

| Variant | arm64 (default) | x86_64 |
|---|---|---|
| 0.8B | **1.1 s** | 1.8 s (58 tok/s generation) |
| 2B | **2.7 s** (prompt processing 104 tok/s) | 5.5–7 s (19 tok/s generation, 69 tok/s prompt) |

Fewer vCPUs are proportionally slower: x86 2B at 5308 MB (4 vCPUs) generates 11 tok/s. Because requests
finish faster, the maximum memory setting costs about the same per request and is the default.

**Documents** are split into chunks of about 512 tokens and processed in parallel, up to
`MaxParallelSegments` (default 8) concurrent Lambda invocations.

**Cold starts** (new execution environment, 2B model):

| Strategy | Cold start | Idle cost |
|---|---|---|
| llama.cpp mmap straight from the lazily loaded image | ~7 min | none |
| **Parallel prefetch of the model file** (default) | ~5–80 s, depending on Lambda's image cache | none |
| SnapStart (`EnableSnapStart=true`) | ~11 s (2.2 s restore) | snapshot cache ≈ USD 0.0000015 per GB-second ≈ USD 40/month at 10240 MB |

The first start of a freshly deployed image is slower still: about 3 min for 0.8B and 7 min for 2B, while Lambda
pulls the image from ECR for the first time. The stack absorbs this with a warm-up invocation before it completes.
Peak memory for the 2B model is about 3.2 GB.

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
uv run cfn-lint deployments/HFLambdaMCP.yaml

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
