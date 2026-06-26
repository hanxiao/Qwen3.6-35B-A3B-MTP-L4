# Qwen3.6-35B-A3B-MTP on NVIDIA L4 24GB

Deploy [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Unsloth **Q4_K_XL** GGUF) with MTP (Multi-Token Prediction) speculative decoding on a **single NVIDIA L4 24 GB** GPU, using the **official llama.cpp Docker image**.

Decode throughput across diverse inputs, measured honestly (greedy, prompt cache disabled, fresh generation each request), **with ECC disabled** (see step 0):

| Workload | Decode tok/s |
|----------|-------------|
| Free-form prose | **~93** |
| Code generation | **~94** |
| JSON / structured | **~93** |
| Chat / dialogue | **~92** |
| Math / reasoning | **~100** |
| Translation (multi) | **~93** |
| Summarization | **~94** |

This is a **~+45 % speedup over the out-of-the-box config (63 tok/s)** — same model, same Q4_K_XL quant, **no quality loss**. Two compounding wins do it: forcing **all MoE experts onto the GPU** (+28 %, see [How the speedup works](#how-the-speedup-works)), and **disabling GDDR6 ECC** (+10 %, see step 0) — ECC silently costs ~10 % of memory bandwidth, and decode here is memory-bound.

> **On the 100 tok/s target:** math/reasoning reaches ~100, but the *minimum* across all input types is ~92 (chat). Pushing every category past 100 is **not reachable losslessly on a single L4**: the raw decode ceiling (no speculation) is ~73 tok/s ECC-off, MTP multiplies it ≤~1.3× (capped by the MoE 8-of-256 expert-union — independently confirmed by an [RTX 3090 benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) where *all* speculative methods go net-negative on this model), and MTP decode is power-bound at the L4's 72 W TDP. See [Why not 100 tok/s](#why-not-100-toks).

---

## Hardware

- GPU: **NVIDIA L4 24 GB** (GCE `g2-standard-8`), ~300 GB/s memory bandwidth, 72 W TDP
- OS: Ubuntu 22.04 Deep Learning VM (`common-cu129` image family — ships NVIDIA driver + CUDA + Docker + nvidia-container-toolkit)
- Instance: `qwen36-mtp-l4`, project `jinaai-dev`, zone `us-west1-a`

## Model

- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~22 GiB, loads to **21.27 GiB** on GPU)
- MoE: 35 B total, 3 B active. The MTP draft head is **baked into the same GGUF** — use the `-MTP-` repo, not the plain GGUF repo.
- At Q4_K_XL the model nearly fills the 24 GB card, which is the whole optimization story below.

---

## Quick start (Docker — recommended)

No source build required. The official `ghcr.io/ggml-org/llama.cpp:server-cuda` image runs MTP out of the box (verified build `9787`, 2026-06).

### 0. Disable ECC (one-time, +10 %, lossless)

The L4 ships with GDDR6 ECC **enabled**, which costs ~10 % of memory bandwidth *and* ~1.5 GB VRAM. Decode here is memory-bandwidth-bound, so disabling ECC is the single highest-ROI system tweak — and it's lossless (ECC corrects rare bit-flips in storage; it does not affect compute correctness). Measured: **raw decode 66 → 73 tok/s, every workload +10 %.**

```bash
sudo nvidia-smi -e 0      # disable ECC (takes effect after reboot)
sudo reboot
# after reboot, verify:
nvidia-smi --query-gpu=ecc.mode.current,memory.total --format=csv,noheader   # -> Disabled, 24570 MiB
```

### 1. Download the model

```bash
mkdir -p ~/models
pip install -q -U "huggingface_hub[hf_xet]"
~/.local/bin/hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --local-dir ~/models/Qwen3.6-35B-A3B-MTP-GGUF
```

### 2. Start the server

**Easiest — prebuilt image** (the optimized config below is baked in as the default command):

```bash
sudo docker run -d --name llama-server --restart unless-stopped \
  --gpus all -p 8080:8080 -v ~/models:/models \
  ghcr.io/hanxiao/qwen3.6-35b-a3b-mtp-l4:latest
```

<details><summary>Or run the official image with the flags explicitly (same result)</summary>

```bash
sudo docker run -d --name llama-server --restart unless-stopped \
  --gpus all -p 8080:8080 \
  -v ~/models:/models \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  --model /models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --alias Qwen3.6-35B-A3B-Q4KXL-MTP \
  --host 0.0.0.0 --port 8080 --jinja --tools all \
  --ctx-size 8192 --parallel 1 \
  --flash-attn on \
  -ngl 99 --n-cpu-moe 0 \
  -ub 64 -b 512 \
  --no-mmap --threads 8 \
  --spec-type draft-mtp --spec-draft-n-max 2
```

</details>

Or use [`docker-compose.yml`](docker-compose.yml): `sudo docker compose up -d`. The prebuilt image is published from [`Dockerfile`](Dockerfile) by [a GitHub Action](.github/workflows/docker-publish.yml).

### 3. Test

```bash
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'
```

---

## The optimized parameters (and why)

| Flag | Value | Why |
|------|-------|-----|
| `-ngl 99 --n-cpu-moe 0` | all layers + **all experts on GPU** | the #1 lever (+28 %). See below. |
| `-ub 64 -b 512` | tiny micro-batch | shrinks the compute buffer by ~1 GB, freeing exactly enough VRAM for `--n-cpu-moe 0` to fit. Decode is batch-1, so this costs nothing at generation time. |
| `--ctx-size 8192` | 8 K | small KV so the full model fits on-GPU. Larger context forces experts back to CPU (see [trade-offs](#context-vs-speed-trade-off)). |
| `--flash-attn on` | — | required; default f16 KV cache (keeps CUDA graphs enabled). |
| `--spec-type draft-mtp --spec-draft-n-max 2` | MTP, 2 drafts | n-max=2 is optimal across workloads (higher values drop accept rate faster than they add tokens). |
| `--no-mmap` | — | loads weights resident in RAM/VRAM instead of mmap (no paging). |

### How the speedup works

The Q4_K_XL weights are 21.27 GiB on a 22.5 GiB-usable card. By default llama.cpp **auto-fit** leaves several MoE expert layers on the CPU to be safe — and every token that routes to a CPU-resident expert pays a large penalty. That caps decode at **63 tok/s**.

The fix: explicitly place **everything on the GPU** (`-ngl 99 --n-cpu-moe 0`) and claw back the VRAM needed to make it fit by **shrinking the compute buffer** (`-ub 64`, since decoding only ever uses batch 1). With zero experts on the CPU, decode jumps to **~81 tok/s (prose)**.

`--n-cpu-moe` sweep (fewer experts on CPU → faster), `-ub 64`, ctx 8192, f16 KV, MTP n-max 2:

| `--n-cpu-moe` | decode tok/s | VRAM |
|---------------|-------------|------|
| auto-fit (default) | 63.3 | 21504 MiB |
| 6 | 63.8 | 19564 MiB |
| 4 | 67.3 | 20404 MiB |
| 2 | 73.5 | 21158 MiB |
| 1 | 76.9 | 21624 MiB |
| **0 (all on GPU)** | **80.9** | 21918 MiB |

---

## Benchmark methodology

Reproducible, no caching tricks:

- **Decode-only metric:** `timings.predicted_per_second` from llama.cpp (excludes prompt processing).
- **No input/output cache:** every request sends `"cache_prompt": false` with distinct prompts — no prefix reuse, no replayed outputs.
- **Greedy** (`temperature: 0`) for stable, reproducible MTP acceptance.
- 256–384 generated tokens per request, averaged over multiple distinct prompts per workload.

Scripts used to produce every number here are in [`scripts/`](scripts/) (`bench.sh`).

### MTP n-max sweep (full GPU residency, prose)

| n-max | tok/s | MTP accept |
|-------|-------|-----------|
| 1 | 75.1 | 0.85 |
| **2** | **80.5** | 0.74 |
| 3 | 79.1 | 0.63 |
| 4 | 77.1 | 0.57 |
| 5 | 72.3 | 0.50 |
| 6 | 69.7 | 0.44 |

### Decode by workload (n-max 2, full GPU residency)

| Workload | tok/s | MTP accept |
|----------|-------|-----------|
| Prose | 77.9 | 0.70 |
| Repetitive/list | 86.1 | 0.83 |
| Code | 87.1 | 0.84 |
| JSON/structured | 88.0 | 0.86 |

---

## Why not 100 tok/s

This was the target; it is not achievable on a single L4 without changing the model, quant, or GPU. The evidence:

1. **Raw decode ceiling = 65.5 tok/s.** `llama-bench` with no speculation, full GPU residency (`-ngl 99 --n-cpu-moe 0`), measures `tg128 = 65.50 ± 0.19 tok/s`. This is the hard memory-bandwidth limit for reading the model's active weights at ~300 GB/s.
2. **MTP multiplies that by only ~1.27–1.37×.** Acceptance is 0.74 on prose (1.27× → 81) and tops out at ~0.86 on JSON (1.37× → 90). Reaching 100 would need a sustained ~1.53× (accept ≈0.92+), which the MTP head does not deliver on any workload tested.
3. **The L4 hits its 72 W power wall under MTP.** Measured power time-series during sustained decode: *raw* (non-MTP) decode is latency-bound — only **32 W**, clock at the **2040 MHz max**, 66 tok/s (batch-1 GEMV can't saturate). Turning on MTP adds the draft+verify work, which pushes the card to **71.6 W (pinned ≥70.5 W in 89% of samples)** and throttles the clock to **~1845 MHz** — i.e. MTP *converts* the latency-bound regime into a power-bound one (that's how it speeds up). The clock can't reach 2040 MHz under load (that would exceed 72 W), so even the counterfactual full-clock ceiling is only ~+10% → **~92 tok/s**, and 72 W is the card's hard max (cannot be raised).
4. **Cross-check:** Unsloth reports ~240 tok/s for this model on an RTX 6000 (Ada, ~960 GB/s). Scaling by bandwidth → 240 × (300/960) ≈ 75 tok/s expected on L4 — consistent with what we measure.
5. **A lower quant barely helps — the L4 is compute/latency-bound, not bandwidth-bound.** Measured Q3_K_XL (16.03 GiB, 25 % smaller than Q4): raw tg128 = **70.8** tok/s (only +8 % over Q4's 65.5, *not* the +33 % pure-bandwidth scaling would predict), and with MTP **93 (prose) / 94 (code) / 98 (JSON)** — **still under 100**. So even dropping to Q3 doesn't clear 100 on the L4, and it isn't worth the quality loss.
6. **Why MTP itself is capped at ~1.3× here (the MoE expert-union).** Qwen3.6-35B-A3B routes **8 of 256 experts per token**. At batch-1 decode, MTP's 1+n_max verified tokens tend to activate *different* experts, so each speculative step pulls in a growing union of expert slices — you'd need a batch of ~94 to amortize the full expert set. That's why higher `--spec-draft-n-max` *reduces* throughput and the net MTP speedup stays near 1.3×. An [independent llama.cpp speculative-decoding benchmark on this exact model (RTX 3090)](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) found *no* variant achieves a net speedup — and the 3090 (936 GB/s, 350 W) is far beefier than the L4.

7. **Kernel-level work can't close the gap either.** A deep investigation (profiling the actual GGML kernels + mining hand-written engines like [antirez/ds4](https://github.com/antirez/ds4) and the [MIT-HAN-Lab kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) methodology) found the bottleneck is the MoE expert matvec `mul_mat_vec_q_moe` (ggml-cuda/mmvq.cu). The one verified *lossless* lever is that MTP (`ncols_dst=2`) disables the gate+up projection fusion, so each runs as a separate launch that re-quantizes the same activation. But re-enabling it is a full kernel **port** (that path has no fusion plumbing), and since the per-token redundant work is small relative to weight reads, the realistic gain is only **+2–5% (~85–92 tok/s), still under 100** — not worth a custom-kernel fork-rebase tax against upstream llama.cpp. ds4's core dp4a matvec is at *parity* with GGML (no headroom). The L4's 72 W power wall, not the kernel, is the binding limit.

**What actually reaches >100 tok/s:** a higher-power / higher-bandwidth GPU — L40S (~864 GB/s), RTX 6000 Ada (~960 GB/s → ~240), or A100 (~2 TB/s). **Not** a quant change (MXFP4 has no FP4 hardware path on Ada/L4 — Blackwell-only; Q3_K_XL tops out at ~98) and **not** a kernel rewrite (ceiling ~92). The single L4's 72 W power wall is the hard limit.

---

## Quality: no regression

- **MTP is distribution-preserving.** A speculative token is accepted only when it matches the target model's own next-token distribution, so MTP changes *speed*, not *what the model produces*. (Greedy output can differ token-for-token from non-speculative decoding because of floating-point differences in how logits are batched during draft verification — but both are equally coherent. Deterministic probes match exactly: `17×23 → 391`, capital of Australia → `Canberra`, reverse "model" → `ledom`.)
- **The speedup touches no quality knob.** Same Q4_K_XL weights, **default f16 KV cache** (higher fidelity than a quantized KV cache), sampling passed per-request. Nothing is traded away for speed — if anything, f16 KV is higher fidelity than the quantized-KV configs used for long-context.

---

## Context vs speed trade-off

The optimized config is tuned for **decode speed at ≤8 K context**. If you need long context instead:

- Raise `--ctx-size` and accept expert offload (`--n-cpu-moe` > 0), trading speed back toward the 63 tok/s baseline.
- For very long context, a quantized KV cache (`--cache-type-k q4_0 --cache-type-v q4_0`) extends maximum context substantially at a further speed cost. This is a different optimization target than this repo's headline (max decode tok/s).

---

## Operations

```bash
# logs / status
sudo docker logs -f llama-server
sudo docker ps

# restart
sudo docker restart llama-server

# decode speed check (greedy, no cache)
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python LRU cache."}],"max_tokens":256,"temperature":0,"cache_prompt":false}' \
  | python3 -c "import sys,json;t=json.load(sys.stdin)['timings'];print(f\"{t['predicted_per_second']:.1f} tok/s\")"

# stop the instance to save cost (correct zone: us-west1-a)
gcloud compute instances stop qwen36-mtp-l4 --project=jinaai-dev --zone=us-west1-a
```

## Lessons learned

1. **Docker works — use it.** The official `ghcr.io/ggml-org/llama.cpp:server-cuda` image runs MTP with `--gpus all` out of the box (no `libllama-common.so.0` issue as of build 9787 / 2026-06). There is **no prebuilt Linux CUDA binary** from llama.cpp (Windows only), so Docker is the prebuilt path; build from source only if you need a flag the image lacks.
2. **Full GPU residency beats auto-fit.** Auto-fit conservatively offloads MoE experts to CPU. `-ngl 99 --n-cpu-moe 0` + a small `-ub` to free the compute buffer is +28 %.
3. **Tiny `-ub` is free at decode time.** Decoding is batch-1, so `-ub 64` doesn't slow generation; it only shrinks the compute buffer (and slows prompt *processing*, which doesn't affect the decode metric).
4. **n-max = 2 is optimal**; higher values drop MTP acceptance faster than they add tokens.
5. **f16 KV keeps CUDA graphs on** (~+3 %) and is higher fidelity than quantized KV. Quantized KV is for *context capacity*, not speed.
6. **Never set sampling parameters server-side.** Global `--temp`/`--top-p`/etc. perturb the draft distribution and tank MTP acceptance; pass sampling per-request.
7. **The L4 is the ceiling.** ~78–90 tok/s is the honest decode limit for Q4_K_XL + MTP on a single, 72 W, 300 GB/s L4. 100 tok/s needs a faster GPU — a lower quant doesn't get there either (measured Q3_K_XL peaks at ~98).
8. **`GGML_CUDA_GRAPH_OPT=1` doesn't help here** (it slightly *hurts*: −1 to −1.5%). It's a lossless CUDA-graph stream-reorder that gives ~40% on RTX 4090/5090, but the gain scales with memory bandwidth the L4 lacks, so the scheduling overhead isn't repaid. It was the only untried lossless lever surviving an exhaustive search (GitHub issues/PRs, source, community benchmarks, arxiv, alt engines, hardware) — confirming no flag/build/runtime change crosses 100 losslessly on this card.
9. **GCE reboots can drop the NVIDIA driver** (kernel auto-upgrades, prebuilt module lags). Fix: `sudo apt-get install -y nvidia-dkms-570-server-open && sudo modprobe nvidia` (DKMS rebuilds for the new kernel).

## Cost

- L4 on-demand: ~$0.81/hr (~$584/mo). Spot: ~$0.24/hr. Stop the instance when idle.
