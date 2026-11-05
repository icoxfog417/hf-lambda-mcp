# Turning Hugging Face Models into Serverless MCP with One Click — Running PLaMo 3 Translate on AWS Lambda + AgentCore Gateway

> This article was translated from the [Japanese original](serverless-hf-mcp.ja.md) by PLaMo 3 Translate 2B running on the very system it describes (`translate_file`, 12 chunks in 8 parallel segments, 64 s), followed by light post-editing by Claude. Code blocks are never sent to the model, so their contents were translated by Claude.

High-performance models like Claude Opus are naturally capable of translation. However, assigning them to translate long documents consumes a massive amount of output tokens, both in terms of time and cost. For tasks like translation—where the "task is clear but the volume is large"—it is logical to delegate them to a small, dedicated model.

To address this, I built a system to **run [PLaMo 3 Translate](https://www.preferred.jp/ja/blog/tech/plamo-3-translate-weight-release), a translation-specialized model released by Preferred Networks (PFN), on AWS Lambda, allowing agents like Claude Code to call it as an MCP tool**. It can be deployed with a single CloudFormation button, and there are no costs while it is not in use.

Furthermore, I did not write code specifically for PLaMo. I consolidated the settings for each model into a single YAML file, ensuring that **the same mechanism can be reused for other models on Hugging Face**.

- Repository: https://github.com/icoxfog417/hf-lambda-mcp
- License: Apache-2.0 (hosting code only; does not include model weights)

This article covers the following three points:

1. What I built and how to use it
2. The underlying mechanism: Architecture and the communication path from local MCP to remote
3. Key considerations and lessons learned: Choosing between ECR and EFS, model-agnostic design, and struggling with cold starts

---

## 1. What I Built: hf-lambda-mcp

### Capabilities

| Tool | Operation Site | Description |
|---|---|---|
| `translate` | AgentCore Gateway (Remote) | Translates text. Allows specifying `target_lang`, `source_lang`, `context` (reference text for determining translations), `glossary` (glossary), and `form` (Japanese style: desu/masu / da/dearu) |
| `translate_file` | Local Client | Translates local `.md` / `.txt` files and saves them as `<filename>.<language>.md` next to the original file |
| `get_model_info` | AgentCore Gateway | Returns the model, revision, and license being used |

The key point of `translate_file` is that **the contents of the file do not pass through the agent's context**. The agent receives only the destination path and metadata. It translates only the main text, without modifying Markdown front matter, code blocks, HTML comments, or math blocks.

### How to Use

**① Deployment**

Open the CloudFormation console via the **Launch Stack** button in the README, review the parameters, and create the deployment.

| Parameter | Default | Description |
|---|---|---|
| `ModelProfile` | `plamo-3-translate` | Profile name under `profiles/` |
| `ModelVariant` | `2b` | `0.8b` (fastest) / `2b` (recommended) / `8b` |
| `Architecture` | `arm64` | 1.5–2.4x faster than x86_64 within the measured range |
| `MaxParallelSegments` | `8` | Number of parallel processes per document |
| `EnableSnapStart` | `false` | Shortens cold starts, but incurs charges even while idle (described below) |

It takes about 20–30 minutes to complete. Model weights are downloaded from Hugging Face **within your account** during deployment.

**② Connect to Your Agent**

Register the stack's output `ClientCommand` directly as an MCP server.

```bash
claude mcp add plamo-translate -- \
  uvx --from git+https://github.com/icoxfog417/hf-lambda-mcp.git@main \
  hf-lambda-mcp https://<gateway-id>.gateway.bedrock-agentcore.us-east-2.amazonaws.com/mcp
```

For authentication, use your AWS credentials (the same ones as the AWS CLI). If the user is not an administrator, grant `bedrock-agentcore:InvokeGateway` permissions to the output `GatewayArn`.

**③ Use**

> Translate docs/guide.md into English with plamo-translate

When you ask this, the agent calls `translate_file(input_path="docs/guide.md", target_lang="en")`, and `docs/guide.en.md` is created.

Here is an example of the actual response (arm64, 2B model, warm state).

```text
translate ja->en        1.9s  "The meeting has been rescheduled for next Monday."
translate en->ja        1.9s  「スタックをデプロイし、エージェントをゲートウェイに接続してください。」
                              (form=polite, glossary: stack→スタック, gateway→ゲートウェイ)
target_lang="EN"        1.2s  isError: invalid arguments: target_lang must use lowercase language codes ...
translate_file (18KB)  54.6s  5 segments in parallel; line structure identical to the source, 0 untranslated lines
```

The Lambda cost for translating the 18KB Markdown file was **approximately $0.024**, including cold start (1,818 GB-seconds).

> ⚠️ **Model License**: PLaMo 3 Translate is distributed under the [PLaMo Community License](https://plamo.preferredai.jp/info/plamo-community-license-en). Commercial use requires registration with PFN. The license for this repository applies only to the hosting code. Please be sure to verify the license for each model yourself.

---

## 2. Under the Hood: Architecture and Communication Paths

### Overview

```
 ┌──────────────┐  stdio   ┌─────────────────┐  HTTPS + SigV4    ┌───────────────────┐  Invoke   ┌──────────────────────────┐
 │ Claude Code  │◄────────►│ hf-lambda-mcp   │◄─────────────────►│ AgentCore Gateway │──────────►│ Lambda (container image) │
 │ etc.         │  (MCP)   │ local client    │ (MCP Streamable   │  (AWS_IAM auth)   │ (Gateway  │  ├ llama-server          │
 └──────────────┘          │ (uvx)           │  HTTP)            └───────────────────┘  IAM role)│  │  127.0.0.1:18080      │
                           └───────┬─────────┘                                                   │  └ GGUF model            │
                                   │                                                             └────────────┬─────────────┘
                                   │ PUT / GET via presigned URLs                                             │ async self-invocation
                                   └────────────────────────►  S3 (uploads/ results/ jobs/ expire in 1 day) ◄─┘  (parallel workers)
```

### Tracing the Communication Path

**① Agent ⇔ Local Client (stdio)**

Claude Code communicates with MCP via standard input/output using `hf-lambda-mcp` launched with `uvx`. This client serves as a replacement for [mcp-proxy-for-aws](https://github.com/aws/mcp-proxy-for-aws) and internally uses that library's `aws_iam_streamablehttp_client` to attach SigV4 signatures.

**② Local Client ⇔ AgentCore Gateway (HTTPS + SigV4)**

Authentication on the Gateway's receiving end is handled by `AWS_IAM`. While OAuth via Cognito was initially considered, we opted for IAM because Cognito does not support the dynamic client registration (DCR / CIMD) required by MCP clients. Since tool names are formatted as `<target-name>___<tool-name>` by the Gateway, the local client removes the prefix and presents them as `translate`.

**③ Gateway ⇔ Lambda (Invoked via IAM Role)**

The Gateway passes the tool's arguments as-is as events to the Lambda. The invoked tool name is stored in `context.client_context.custom["bedrockAgentCoreToolName"]`. The tool schema is generated from a profile during build and stored in S3; the target definition then references that S3 URI.

**④ Inside Lambda**

During the Lambda initialization phase, `llama-server` from llama.cpp is launched as a child process, and the handler calls the OpenAI-compatible API at `127.0.0.1:18080`. Port 8080 was avoided because it conflicts with the Lambda emulator (RIE) running locally.

**⑤ File Translation Path**

When `translate_file` is called, the process proceeds as follows:

1. `create_upload` retrieves a presigned PUT URL to S3, and the local client uploads the file directly (the user does not need S3 permissions or a bucket).
2. Calling `start_job` causes Lambda to asynchronously call itself to spawn "planning" workers.
3. The planning worker splits the document and spawns up to 8 "segment" workers in parallel.
4. Each worker saves its progress to S3 after processing one chunk. When the remaining time runs low, it saves its state and calls itself again, allowing it to process data even if the duration exceeds Lambda's 15-minute limit.
5. The last worker to complete assembles the entire result and saves it to `results/`.
6. The local client polls `get_job` to send MCP progress notifications, and upon completion, downloads and saves the result via the presigned GET URL.

### Deployment Flow (Details of the One-Click Process)

```
CloudFormation
 ├─ Create S3 / ECR / CodeBuild / build-coordinator Lambda
 ├─ Custom resource ModelImage ─► start CodeBuild (pass the CloudFormation response URL as an env var)
 │     CodeBuild: clone repo → resolve profile → download model from Hugging Face
 │               → build llama.cpp → push image to ECR → upload tool schema to S3
 │     EventBridge (build finished) ─► coordinator Lambda ─► return SUCCESS / FAILED to CloudFormation
 ├─ Lambda function / version / alias `live`
 ├─ Custom resource WarmUp ─► invoke the new version once (fail the deployment if the model cannot load)
 └─ AgentCore Gateway (AWS_IAM) + Lambda target
```

In the [ndl-ocr-lite-lambda](https://github.com/icoxfog417/ndl-ocr-lite-lambda) I referenced, CDK was executed within CodeBuild. Now that **AgentCore Gateway is available as a native CloudFormation resource** (`AWS::BedrockAgentCore::Gateway` / `GatewayTarget`), I've consolidated everything into a single template. Since build success/failure is returned to CloudFormation via EventBridge, the stack is rolled back with a link to the logs regardless of which phase fails.

---

## 3. Key Takeaways and Lessons Learned

### 3.1 Model-Agnostic Design: "Don't Write the Model Card Twice"

I want to avoid creating separate repositories for every Hugging Face model. However, making the design too generic would make the tool difficult for agents to use. Therefore, I drew a line: **read what can be mechanically gleaned from the model's artifacts, and only write what cannot be read in the profile**.

| Required Information | Source | Write in Profile |
|---|---|---|
| Prompt format | `chat_template.jinja` (or templates embedded in GGUF). Passed directly to llama-server via `--jinja` | Not required |
| Parameter names | Variables referenced by the chat template (`target_lang`, `glossary`, etc.) | Not required (matched during build) |
| Input validation | `raise_exception()` within the template | Not required |
| Context length | GGUF metadata | Not required |
| License | Hugging Face model card metadata | Not required |
| Tool description and types for agents | Model card body (human-facing text) | **Required** |
| Which file size/quantization to use | Decision made at deployment time | **Required** |
| Document processing settings | Platform features | **Required** |

The PLaMo profile ended up consisting almost entirely of the tool description.

```yaml
name: plamo-3-translate
source:
  repo: pfnet/plamo-3-translate-{variant}-GGUF
  file: plamo-3-translate-{variant}-Q4_K_M.gguf
  variants: [0.8b, 2b, 8b]
  default_variant: 2b
generation: {temperature: 0}
tools:
  - name: translate
    description: Translate text with PLaMo 3 Translate ...
    input: {name: text, description: Source text to translate}
    params:            # names are checked against the chat template's variables
      target_lang: {type: string, required: true, description: ...}
      glossary:    {type: array, description: ..., items: {...}}
      form:        {type: string, enum: [plain, polite], description: ...}
    document: {formats: [txt, md], max_chunk_tokens: 512, history: 1,
               output_name: "{stem}.{target_lang}{ext}"}
```

There are three points that I felt worked well:

- **Input validation can be left to the model**: The PLaMo template automatically validates conditions such as the language code format and whether `form` can only be specified for Japanese output. If I returned those error messages as tool errors, the agent would read them and fix the arguments on its own. This saved me from writing a single line of validation code.
- **No response configuration required**: We debated whether to define the response format, but the assistant's response text alone was sufficient for output. While some model card examples specify PLaMo-specific control tokens as additional stop tokens, in practice, it stopped correctly at the EOS token, making such configuration unnecessary.
- **Extracting template variables requires a bit of a workaround**: The results of `jinja2.meta.find_undeclared_variables` include variables that are being `set` locally by macros within the template. To address this, I added a process to exclude assigned variables while treating variables that check "whether they are defined" (e.g., `{% if target_lang is not defined %}`) as arguments passed from the caller.

### 3.2 ECR vs. EFS: The real constraint isn't image size

The reference implementation placed its ONNX models on EFS. Again, I was concerned that "since the container image limit is 10GB, large models might not fit." However, upon closer inspection, the bottleneck was elsewhere:

- **Lambda's memory limit is also 10GB**: The model and KV cache must be loaded into memory. In other words, a model that doesn't fit in a 10GB image will not run in Lambda even if placed on EFS.
- **Non-model components are small**: Even including all CPU variants, llama.cpp was **71MB**. Even including the Python base image, the non-model portion is about 0.65GB.
- **Smaller models are more practical**: CPU inference speed is roughly inversely proportional to model size. Models of 3–5GB or less are what are actually usable.
- **EFS has other costs**: It requires a VPC and a mechanism to deploy the model to EFS. Additionally, every cold start reads the model over the network.

In conclusion, I opted to **bake the model into the image and store it in ECR**. I believe that if I want to use larger models, the proper approach would be to connect a runtime capable of using GPUs (such as AgentCore Runtime or SageMaker) behind the same Gateway, rather than using EFS.

### 3.3 Battling Cold Starts: 415 Seconds → A few dozen seconds

When I first measured it, the cold start for the 2B model was **415 seconds**.

The cause was a bad fit between Lambda's lazy loading of container images (fetching chunks on demand) and llama.cpp's mmap (reading pages in scattered order as needed). Every random page fault ran straight into the latency of lazy loading.

| Solution | Cold Start | Idle Cost |
|---|---|---|
| Do nothing (load directly via mmap) | Approx. 415 seconds | None |
| **Prefetch model files sequentially across multiple threads before starting llama-server** (Default) | **A few seconds to 80 seconds** (depending on Lambda's image cache state) | None |
| SnapStart (restore from a snapshot after model loading) | Approx. 11 seconds (restoration itself takes 2.2 seconds) | **Around $40/month with a 10GB setting** |

I learned three things here:

- **SnapStart now supports container images, but you're charged while idle**: It allows restoring the entire llama-server child process from a snapshot, and the effect was dramatic. However, for Python images, snapshot cache fees ($0.0000015046 per GB-second) apply. With a 10GB memory setting, it costs around $40/month even when not in use. Since this conflicts with the goal of "zero cost when idle," I made it opt-in.
- **Simple prefetching works wonders**: Just read model files in 16MB increments sequentially across 8 threads before mmap. This alone reduced 415 seconds to a few dozen seconds (the implementation is about 20 lines).
- **Initial startup immediately after deployment is particularly slow**: The first launch of a newly pushed image took about 7 minutes on 2B because it had to re-fetch from ECR. I therefore changed it to call a new version once via a CloudFormation custom resource before exposing the Gateway. This absorbs the delay during deployment and also serves as a **smoke test** that fails the deployment if the model cannot be loaded.

### 3.4 The Reality of Inference on Lambda's CPU and the Answer: arm64

Lambda does not have GPUs. Allocating 10GB of memory results in 6 vCPUs. The measurement results are as follows (Q4_K_M, English translation of a 250-character Japanese paragraph, warm state):

| Model | arm64 | x86_64 |
|---|---|---|
| 0.8B | **1.1 seconds** | 1.8 seconds (generation 58 tok/s) |
| 2B | **2.7 seconds** (prompt processing 104 tok/s) | 5.5–7 seconds (generation 19 tok/s, prompt processing 69 tok/s) |

- On x86, speed remained nearly constant regardless of the number of threads, Flash Attention, or batch size. The computational performance of CPUs without AVX-512 appears to be the bottleneck.
- Since **arm64 (Graviton) is 1.5–2.4 times faster and costs about 20% less per GB-second**, I set the default to arm64. Note that GCC 11 on Amazon Linux 2023 cannot compile the armv9.2 variant of llama.cpp. For arm64, I only build one version: `armv8.2-a+dotprod+fp16`.
- Reducing memory also reduces vCPUs and slows down performance. Since the per-request cost remains nearly the same, I set the maximum of 10GB as the default.

### 3.5 Parallelization: 305 seconds → 55 seconds

On CPU, the 2B model takes approximately 55 seconds to translate a 1,024-token chunk. Processing an 18KB Markdown file (3 chunks) sequentially took **305 seconds**.

To address this, I reduced the chunks to 512 tokens and processed them in parallel using up to 8 Lambda instances. The result is **54.6 seconds** (arm64, 5 segments, including cold start). Since the total computational cost is the same, the cost remains nearly unchanged, and only the wait time is reduced.

I also implemented some optimizations for consistency.

- **Pass the original text and translation of the previous chunk as chat history within the segment**: Since PLaMo's templates can accept "pairs of original text and translation" as history, I was able to achieve this without writing PLaMo-specific code.
- **Extend the history without shifting it**: If you shift the history to only pass the most recent pair, the beginning of the prompt changes every time, preventing llama-server from reusing the previous computation results (KV cache). I increased the cache reuse rate by continuing to extend the history until it fit within half the context, then restarting from the most recent pair once that limit was exceeded.
- **Use `glossary` to unify terms that span segment boundaries**. If you prioritize absolute consistency, set `MaxParallelSegments=1` to process the entire sequence in order.

### 3.6 The "Delegation Makes It Cheaper" Pitfall

In the initial design, the agent was expected to pass the main text to `translate(text=...)`. However, upon further reflection, **writing the main text as an argument itself consumes the agent's output tokens by the length of the original text**. This makes it essentially no different from the agent translating on its own.

To make it truly cheap, you need to **pass the location of the data** rather than the data itself. However, a Lambda behind a Gateway cannot see the user's disk, and `mcp-proxy-for-aws` only relays messages. Therefore, I built a custom local client that operates as follows:

- Relay remote tools as-is, while **automatically generating** a derivative tool named `<tool-name>_file` for tools that handle document processing.
- Determine which tool is the target via the remote `get_model_info`, ensuring the client is model-independent.
- Transfer files to and from the S3 bucket created by the stack using presigned URLs.

### 3.7 Small but Painful Pitfalls

- **Lambda runs as a non-root user**: Files retrieved via `ADD <URL>` in the Dockerfile had permissions set to 0600, making them unreadable. Since the local emulator runs as root, this issue goes unnoticed.
- **anyio's cancel scope**: Opening an MCP session to a remote endpoint in one request processing task and then closing it in another task results in an exception. I modified the code to open and close sessions within the same task for each tool call. This also had the side benefit of refreshing the credentials every time.
- **Function default arguments**: Writing `def plan_segments(n, max_parallel=MAX_PARALLEL)` binds the value at the time the function is defined. This was discovered when swapping the number of parallel processes in tests had no effect.

---

## Summary and Future Work

- I built a system to turn Hugging Face's GGUF models into MCP tools using a single YAML file and a one-click deployment. The first example is PLaMo 3 Translate.
- There is no idle cost; short sentences take about 2 seconds, and an 18KB document takes about 1 minute and costs approximately $0.024 to translate.
- The key to generalizability was drawing a line: using the model artifacts (chat templates and GGUF) as the source of configuration, while writing only the elements that require human judgment in the profile.

Future goals:

- Support for `.docx` and `.html` (just one more splitter; no profile changes needed)
- `hf-lambda-mcp init <repo>`: A command to automatically generate profile templates from chat templates and GGUF
- Lining up multiple models behind a single Gateway, and a GPU backend for larger models

Beyond translation, Hugging Face has many "small models with clear areas of expertise." Frontier models act as coordinators, while small serverless models handle the actual execution. I hope this serves as a catalyst for experimenting with this kind of division of labor.
