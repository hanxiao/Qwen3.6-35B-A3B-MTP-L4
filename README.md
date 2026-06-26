# Qwen3.6-35B-A3B-MTP on NVIDIA L4 24GB

Deploy [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (Unsloth **Q4_K_XL** GGUF) with MTP (Multi-Token Prediction) speculative decoding on a **single NVIDIA L4 24 GB** GPU, using the **official llama.cpp Docker image**.

Decode throughput across diverse inputs — measured honestly (greedy, prompt cache disabled, a fresh generation per request), **with ECC disabled** (see [step 0](#0-disable-ecc-one-time-10-lossless)):

| Workload | Decode tok/s |
|----------|-------------|
| Free-form prose | **~93** |
| Code generation | **~94** |
| JSON / structured | **~93** |
| Chat / dialogue | **~92** |
| Math / reasoning | **~100** |
| Translation (multilingual) | **~93** |
| Summarization | **~94** |

That's **~+45 % over the out-of-the-box config (63 tok/s)** — same model, same Q4_K_XL quant, **no quality loss**. Two compounding, lossless wins do it:

1. **Full GPU residency** — force all MoE experts onto the GPU (+~28 %, [how](#how-the-two-wins-were-measured)).
2. **Disable GDDR6 ECC** — it silently costs ~10 % of memory bandwidth, and decode here is memory-bound (+~10 %, [step 0](#0-disable-ecc-one-time-10-lossless)).

> **On the 100 tok/s target:** math/reasoning reaches ~100, but the *minimum* across all input types is ~92 (chat). Getting **every** category over 100 is **not achievable on a single L4** with this model+quant — proven by measurement, not assumption (see [Why not 100](#why-not-100-toks)). It needs a higher-bandwidth GPU; a lower quant actually makes it *slower*.

---

## Hardware

- GPU: **NVIDIA L4 24 GB** (GCE `g2-standard-8`), ~300 GB/s memory bandwidth, 72 W TDP, Ada `sm_89`
- OS: Ubuntu 22.04 Deep Learning VM (`common-cu129` family — ships NVIDIA driver + CUDA + Docker + nvidia-container-toolkit)
- Reference instance: `qwen36-mtp-l4`, project `jinaai-dev`, zone `us-west1-a`

## Model

- [unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), file `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~22 GiB; ~21.3 GiB resident on the GPU).
- MoE: 35 B total, 3 B active (8 of 256 experts per token). The MTP draft head is **baked into this GGUF** — use the `-MTP-` repo, not the plain GGUF repo.
- At Q4_K_XL the weights nearly fill the 24 GB card — that tight fit is the whole optimization story below.

---

## Quick start (Docker)

No source build needed. The official `ghcr.io/ggml-org/llama.cpp:server-cuda` image runs MTP out of the box with `--gpus all` (verified mid-2026; the tag tracks latest master).

### 0. Disable ECC (one-time, +~10 %, lossless)

The L4 ships with GDDR6 ECC **enabled**, costing ~10 % of memory bandwidth *and* ~1.5 GB VRAM. Decode here is memory-bandwidth-bound, so this is the single highest-ROI tweak — and it's lossless (ECC corrects rare stored-bit flips; it does not affect compute). Measured: **raw decode 65.5 → 73 tok/s; every workload +~10 %.**

```bash
sudo nvidia-smi -e 0      # disable ECC (applies after reboot)
sudo reboot
# verify after reboot:
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

<details><summary>Or run the official image with the flags explicit (identical result)</summary>

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

Or [`docker-compose.yml`](docker-compose.yml): `sudo docker compose up -d`. The prebuilt image is built from [`Dockerfile`](Dockerfile) by [a GitHub Action](.github/workflows/docker-publish.yml) on every change.

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
| `-ngl 99 --n-cpu-moe 0` | all layers + **all experts on GPU** | the #1 lever (+~28 %). See below. |
| `-ub 64 -b 512` | tiny micro-batch | shrinks the compute buffer by ~1 GB, freeing exactly enough VRAM for `--n-cpu-moe 0` to fit. Decode is batch-1, so this costs nothing at generation time. |
| `--ctx-size 8192` | 8 K | small KV so the full model fits on-GPU. Larger context forces experts back to CPU (see [trade-offs](#context-vs-speed-trade-off)). |
| `--flash-attn on` | — | required; keeps the default f16 KV cache (which keeps CUDA graphs enabled). |
| `--spec-type draft-mtp --spec-draft-n-max 2` | MTP, 2 drafts | n-max=2 is optimal across workloads (higher values drop acceptance faster than they add tokens). |
| `--no-mmap --threads 8` | — | weights resident in RAM/VRAM (no paging); 8 vCPUs. |

### How the two wins were measured

The Q4_K_XL weights are ~21.3 GiB on a ~22.5 GiB-usable card. By default llama.cpp **auto-fit** parks several MoE expert layers on the CPU to be safe, and every token routed to a CPU-resident expert pays a big penalty — capping decode at **~63 tok/s**.

**Win 1 — full residency.** Place everything on the GPU (`-ngl 99 --n-cpu-moe 0`) and reclaim the VRAM to fit it by shrinking the compute buffer (`-ub 64`). Sweeping `--n-cpu-moe` down (measured with ECC still on, to isolate this lever):

| `--n-cpu-moe` | decode tok/s | VRAM |
|---------------|-------------|------|
| auto-fit (default) | 63.3 | 21504 MiB |
| 4 | 67.3 | 20404 MiB |
| 2 | 73.5 | 21158 MiB |
| 1 | 76.9 | 21624 MiB |
| **0 (all on GPU)** | **80.9** | 21918 MiB |

**Win 2 — ECC off** then adds a uniform **~+10 %** on top, yielding the headline ~92–100 tok/s. The two compound to ~+45 % overall.

---

## Benchmark methodology

Reproducible, no caching tricks:

- **Decode-only metric:** `timings.predicted_per_second` from llama.cpp (excludes prompt processing).
- **No input/output cache:** every request sends `"cache_prompt": false` with distinct prompts — no prefix reuse, no replayed outputs.
- **Greedy** (`temperature: 0`) for stable, reproducible MTP acceptance.
- 256 generated tokens/request, averaged over 7 distinct workload types (prose, code, JSON, chat, math, multilingual, summarization).

The benchmark script is [`scripts/bench.sh`](scripts/bench.sh) (point it at a running server).

### MTP n-max sweep (full GPU residency)

n-max=2 is the sweet spot — higher values verify more draft tokens but acceptance falls faster than throughput rises (the MoE expert-union effect, see [Why not 100](#why-not-100-toks)):

| n-max | tok/s | MTP accept |
|-------|-------|-----------|
| 1 | 75 | 0.85 |
| **2** | **81** | 0.74 |
| 3 | 79 | 0.63 |
| 4 | 77 | 0.57 |
| 6 | 70 | 0.44 |

*(measured ECC-on, isolating n-max; ECC-off scales all rows ~+10 %.)*

---

## Why not 100 tok/s

Getting **every** input type over 100 is not achievable on a single L4 with this model+quant, and I proved it by implementing and measuring — not by hand-waving:

1. **Raw decode ceiling ≈ 73 tok/s (ECC-off).** `llama-bench`, no speculation, full GPU residency: `tg ≈ 73 tok/s`. This is the memory-read limit for the model's ~1.7 GB of active weights per token at ~300 GB/s. (It was 65.5 with ECC on — that's what step 0 fixes.)
2. **MTP multiplies that by only ~1.27–1.37×.** Acceptance is ~0.77 on prose (1.27× → ~93) and tops out ~0.90 on math (1.37× → ~100). Pushing chat (~0.77 → ~92) past 100 would need a sustained ~1.4×+, which the MTP head doesn't deliver on lower-predictability content.
3. **MTP decode is power-bound at 72 W.** Time-series during sustained decode: raw decode is latency-bound (~27–32 W, clock at the 2040 MHz max). MTP adds draft+verify work that pins the card at **71.6 W (≥70.5 W in 89 % of samples)** and throttles the clock to ~1845 MHz. Even the counterfactual full-clock would be only ~+10% → ~95, and 72 W is the card's hard max.
4. **The MoE expert-union caps MTP at ~1.3×.** With 8-of-256 routing, MTP's `1+n_max` verified tokens activate *different* experts, so each speculative step pulls in a growing union of expert slices (you'd need a batch of ~94 to amortize the full set). An [independent benchmark on this exact model (RTX 3090, 3× the L4's bandwidth)](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) finds *all* speculative methods go net-negative — confirming the cap is architectural, not L4-specific.
5. **A lower quant *backfires* (measured).** Q3_K_XL + ECC-off scores **MIN 86.9 tok/s — below Q4's 91.7** — because the coarser quant degrades the MTP draft head (acceptance 0.766 → 0.714), and the lost speculative speedup outweighs the smaller (16 vs 22 GiB) weight reads. So a lower quant loses on **both** quality and speed here.
6. **Kernel work can't close it either.** Profiling + mining hand-written engines ([antirez/ds4](https://github.com/antirez/ds4), [MIT-HAN-Lab kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents)) showed the only lossless lever is fusing the MoE gate+up on the MTP-verify path (upstream leaves it unfused). I [implemented it](#advanced-moe-verify-fusion-patch-optional-1) and **measured +1 %** — bounded by the ~3 % per-token launch-overhead budget (measured via CUDA-graphs on/off). Megakernel approaches (Mirage etc.) don't apply: their win is cross-layer weight prefetch, but our weights are already fully VRAM-resident.

**What actually reaches >100:** a higher-bandwidth GPU — L40S (~864 GB/s), RTX 6000 Ada (~960 GB/s → ~240), or A100 (~2 TB/s). Cross-check: Unsloth reports ~240 tok/s on RTX 6000; scaling by bandwidth → ~75 on L4, consistent with what we measure. **Not** a quant change and **not** a kernel rewrite. On the single L4, **~92–93 (min) is the wall** for this model.

---

## Quality: no regression

- **MTP is distribution-preserving.** A speculative token is accepted only when it matches the target model's own next-token distribution — MTP changes *speed*, not *what the model produces*. (Greedy output can differ token-for-token from non-speculative decoding due to floating-point batching differences in draft verification, but both are equally coherent; deterministic probes match exactly: `17×23 → 391`, capital of Australia → `Canberra`, reverse "model" → `ledom`.)
- **The speedup touches no quality knob.** Same Q4_K_XL weights, default **f16 KV cache** (higher fidelity than a quantized KV cache), sampling passed per-request. ECC-off is lossless. Nothing is traded for speed.

## Advanced: MoE verify-fusion patch (optional, +~1 %)

[`patches/moe-verify-fusion.patch`](patches/moe-verify-fusion.patch) is an experimental, **lossless** ggml-cuda patch that fuses the gate+up projections (and SwiGLU) of the MoE FFN on the **MTP verify path** (`MUL_MAT_ID` with `ncols_dst > 1`), which upstream runs unfused (the single-token path is already fused upstream).

Measured A/B (same source build, ECC-off): **+1.0–1.1 %** (chat 82.8 → 83.6 tok/s), output bit-equivalent. A genuine but small win — bounded by the ~3 % launch-overhead budget, which is why it can't reach 100. It requires a source build (`cmake --build`), so it's *not* in the prebuilt Docker image; it's upstreamable. Apply with `patch -p1 < patches/moe-verify-fusion.patch`.

---

## Context vs speed trade-off

The optimized config targets **decode speed at ≤8 K context**. For long context instead:

- Raise `--ctx-size` and accept expert offload (`--n-cpu-moe` > 0), trading speed back toward the ~63 tok/s baseline.
- For very long context, a quantized KV cache (`--cache-type-k q4_0 --cache-type-v q4_0`) extends maximum context at a further speed cost — a different optimization target than this repo's (max decode tok/s).

---

## Operations

```bash
sudo docker logs -f llama-server      # logs
sudo docker ps                        # status
sudo docker restart llama-server      # restart

# decode-speed check (greedy, no cache)
curl -s http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a Python LRU cache."}],"max_tokens":256,"temperature":0,"cache_prompt":false}' \
  | python3 -c "import sys,json;t=json.load(sys.stdin)['timings'];print(f\"{t['predicted_per_second']:.1f} tok/s\")"

# stop the instance to save cost
gcloud compute instances stop qwen36-mtp-l4 --project=jinaai-dev --zone=us-west1-a
```

## Lessons learned

1. **Use the Docker image.** The official `ghcr.io/ggml-org/llama.cpp:server-cuda` runs MTP with `--gpus all` out of the box. There is no prebuilt *Linux* CUDA binary from llama.cpp (Windows only), so Docker is the prebuilt path; build from source only for a flag the image lacks.
2. **Full GPU residency beats auto-fit** (+~28 %): `-ngl 99 --n-cpu-moe 0` plus a small `-ub` to free the compute buffer.
3. **Tiny `-ub` is free at decode time** (batch-1); it only shrinks the compute buffer and slows *prompt processing*, not decode.
4. **n-max = 2 is optimal**; higher values drop MTP acceptance faster than they add tokens.
5. **f16 KV keeps CUDA graphs on** and is higher fidelity than a quantized KV cache. Quantized KV is for *context capacity*, not speed (and it lowers MTP acceptance).
6. **Never set sampling parameters server-side.** Global `--temp`/`--top-p`/etc. perturb the draft distribution and tank MTP acceptance; pass sampling per-request.
7. **ECC-off is the biggest single lever here** (+~10 %, lossless) — decode is memory-bound, and GDDR6 ECC taxes bandwidth. Don't over-focus on the GPU cores; the memory subsystem mattered most.
8. **A lower quant is counter-productive** for this MTP model — Q3_K_XL measured *slower* than Q4 because it degrades the draft head. `GGML_CUDA_GRAPH_OPT=1` also slightly *hurts* on the L4 (its win scales with bandwidth the L4 lacks).
9. **The L4 is the ceiling**: ~92–100 tok/s (min ~92) is the honest limit for Q4_K_XL + MTP on one 72 W / 300 GB/s L4. >100-on-everything needs a faster GPU.
10. **GCE reboots can drop the NVIDIA driver** (kernel auto-upgrades, prebuilt module lags). Fix: `sudo apt-get install -y nvidia-dkms-570-server-open && sudo modprobe nvidia`.

## Cost

L4 on-demand ~$0.81/hr (~$584/mo); spot ~$0.24/hr. Stop the instance when idle.
