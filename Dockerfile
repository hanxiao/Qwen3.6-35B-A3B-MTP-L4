# Decode-optimized llama.cpp server for Qwen3.6-35B-A3B Q4_K_XL + MTP on a single NVIDIA L4 24GB.
#
# Thin wrapper over the official llama.cpp CUDA server image: same validated binary, with the
# decode-optimized launch config baked in as the default command (full GPU residency via
# `-ngl 99 --n-cpu-moe 0` + a shrunk `-ub 64` compute buffer). With ECC disabled (README step 0),
# measured ~91 tok/s (chat) to ~99 (math), greedy, no prompt cache. See README for full benchmark + analysis.
#
# Defaults to --ctx-size 56320 (the measured max before OOM on a 24 GB L4). This REQUIRES ECC off
# (README step 0) — on an ECC-on card it will OOM at load; drop --ctx-size to fit if you can't disable ECC.
#
# The 22 GiB model is NOT baked in — mount it (README step 1):
#   docker run -d --gpus all -p 8080:8080 \
#     -v ~/models:/models \
#     ghcr.io/hanxiao/qwen3.6-35b-a3b-mtp-l4:latest
#
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

LABEL org.opencontainers.image.source="https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4"
LABEL org.opencontainers.image.description="Qwen3.6-35B-A3B Q4_K_XL + MTP, decode-optimized for a single NVIDIA L4 24GB (full GPU residency; ~91-99 tok/s with ECC off). Mount the GGUF at /models — see repo README."
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Inherits ENTRYPOINT ["/app/llama-server"] from the base image; this is the default arg set.
CMD ["--model", "/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", \
     "--alias", "Qwen3.6-35B-A3B-Q4KXL-MTP", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--jinja", "--tools", "all", \
     "--ctx-size", "56320", "--parallel", "1", \
     "--flash-attn", "on", \
     "-ngl", "99", "--n-cpu-moe", "0", \
     "-ub", "64", "-b", "512", \
     "--no-mmap", "--threads", "8", \
     "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
