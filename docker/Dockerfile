# Lambda container image: llama.cpp (llama-server) + model artifacts + hf_lambda_mcp server.
#
# Build context: repository root, after running `hf-lambda-mcp-build ... --out build`, which
# writes build/runtime.json, build/tool-schema.json and build/model/{model.gguf,chat_template.jinja}.
#
#   docker buildx build --platform linux/amd64 --provenance=false -f docker/Dockerfile -t <repo>:<tag> .
#
# Layers are ordered by how often they change: llama.cpp < model < code.

ARG BASE_IMAGE=public.ecr.aws/lambda/python:3.12

FROM ${BASE_IMAGE} AS llama
RUN dnf install -y gcc-c++ cmake git make && dnf clean all
ARG LLAMA_CPP_REF=v0.9.0
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
# GGML_BACKEND_DL + GGML_CPU_ALL_VARIANTS: build every CPU micro-architecture variant and pick
# the best one at runtime, because Lambda does not guarantee a specific CPU generation.
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=ON \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_INSTALL_RPATH='$ORIGIN' -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
 && cmake --build build --target llama-server -j"$(nproc)"
RUN mkdir -p /out \
 && cp build/bin/llama-server /out/ \
 && cp -P build/bin/*.so* /out/ \
 && cp -L /usr/lib64/libgomp.so.1 /out/ \
 && ldd /out/llama-server | (! grep "not found")

FROM ${BASE_IMAGE}
# Allow SnapStart: the model is loaded during init and captured in the snapshot.
LABEL com.amazonaws.lambda.feature.snapstart="Allow"
COPY --from=llama /out/ /opt/llama/
COPY build/model/ /opt/model/
COPY build/runtime.json /opt/model/runtime.json
COPY src/hf_lambda_mcp/ ${LAMBDA_TASK_ROOT}/hf_lambda_mcp/
ENV MODEL_DIR=/opt/model LLAMA_SERVER_BIN=/opt/llama/llama-server
CMD ["hf_lambda_mcp.server.handler.handler"]
