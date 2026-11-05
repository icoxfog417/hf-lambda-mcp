# CLAUDE.md

## Mission

Serve small Hugging Face models as MCP tools on AWS Lambda behind Amazon Bedrock AgentCore Gateway,
so frontier agents can delegate bulk work cheaply. One model = one YAML profile in `profiles/`.
The first profile is PLaMo 3 Translate. See `README.md` and `spec/design.md`.

## Principles

- **The server is model-agnostic.** Never add model-specific branches to `src/`. Anything specific
  goes in a profile, and the profile holds only what the Hugging Face artifacts cannot provide. Prompt
  format, parameter names and validation come from the chat template; context length comes from the GGUF metadata.
- **The Lambda runtime uses only the standard library plus boto3** (`server/`, `documents.py`). Build-time
  dependencies (`yaml`, `jinja2`, `gguf`, `huggingface_hub`) are imported lazily in `profile.py`.
- **File contents never pass through the agent.** The `*_file` tools exchange data via presigned S3 URLs.
- Keep the one-click template self-contained (`deployments/template.yaml`, buildspec inlined).

## Layout

```
profiles/                 model profiles (YAML)
src/hf_lambda_mcp/
  profile.py              profile loading/validation, HF resolution, Gateway schema (build CLI)
  documents.py            txt/md parsing, chunking, reassembly (stdlib only)
  server/                 Lambda: handler, llama-server manager, engine, S3 jobs
  client/cli.py           local stdio MCP client (uvx hf-lambda-mcp)
docker/Dockerfile         Lambda image (llama.cpp + model + server)
deployments/template.yaml one-click CloudFormation (CodeBuild image build + Lambda + Gateway)
tests/                    pytest (no AWS needed; moto for S3)
.sandbox/                 experiments, gitignored
```

## Commands

Always run Python through uv.

```bash
uv sync --all-extras
uv run pytest
uv run cfn-lint deployments/template.yaml
uv run --extra build hf-lambda-mcp-build profiles/plamo-3-translate.yaml --variant 0.8b --out build
```

## Gotchas learned the hard way

- Lambda runs as a non-root user: files in the image must be world-readable (`chmod -R a+rX build`).
- Without SnapStart, cold starts of multi-GB images take minutes (lazy image loading + mmap).
- llama-server must not listen on port 8080 (the Lambda RIE uses it locally); the default is 18080.
- Gateway tool names arrive as `<target>___<tool>` in `context.client_context.custom['bedrockAgentCoreToolName']`.
- Test deployments go to **us-east-2**.
