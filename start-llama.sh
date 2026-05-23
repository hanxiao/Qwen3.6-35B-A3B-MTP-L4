#!/bin/bash
export LD_LIBRARY_PATH=/home/hanxiao/llama.cpp/build/bin
exec /home/hanxiao/llama.cpp/build/bin/llama-server \
  --model /home/hanxiao/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias Qwen3.6-35B-A3B-Q4KXL-MTP \
  --ctx-size 8192 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --chat-template-kwargs '{"enable_thinking":true}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --jinja \
  --tools all \
  --host 0.0.0.0 \
  --port 8080
