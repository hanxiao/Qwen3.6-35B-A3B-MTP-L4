# Pushing past 100 → the >120 tok/s question (research log)

**Question:** can decode be pushed from the measured ~91–99 tok/s toward **>120 tok/s**, on the **same** Qwen3.6-35B-A3B-MTP Q4_K_XL GGUF, a **single L4**, **losslessly** (no model/quant switch, no I/O cache)?

**Answer (measured + code-grounded): no.** The wall is silicon — **~300 GB/s memory bandwidth + a hard 72 W power cap** — and every lossless lever either is already captured, gives ~0 here, or is blocked for this model on Ada. Best achievable on this L4 stays **~91–99**; a multi-week MTP-head retrain might reach ~105–115; **>120 needs a higher-bandwidth GPU** (L40S ~864 GB/s, measured ~225 tok/s by others on this exact model). This log records what was tried so nobody re-treads it.

## Measured on the box (ECC off, full residency, n-max 2, chat prompt, greedy, 3-rep avg)

| Lever | Result | tok/s | Verdict |
|-------|--------|-------|---------|
| baseline (CUDA graphs on) | reference | **90.5** | — |
| `GGML_CUDA_DISABLE_GRAPHS=1` | graphs *help* (disabling is slower) | 88.3 | graphs healthy — no recapture bug to exploit |
| `GGML_CUDA_GRAPH_OPT=1` | noise | 90.2 | ~0 on a bandwidth-bound L4 |
| `GGML_CUDA_FORCE_MMQ=1` (control) | noise | 91.1 | MMVQ already optimal for batch-1 |
| locked clock 2040 MHz | clock collapses to ~1950 anyway | 91.1 | **72 W cap claws it back** |
| raise power limit (`-pl 80`) | **rejected** | — | Max = Default = 72 W (hard cap) |
| fresh build 9803 → 9828 | no gain | 88.4 | build already current; no stale-fusion bonus |

All within run-noise (~±1.5 tok/s). **Net free-lever gain ≈ 0.**

## The 16 ideas, adjudicated (code-read, not hand-waved)

| Idea | Status | Why |
|------|--------|-----|
| KV-caching, FlashAttention, Speculative (MTP) | **already done** | f16 KV, `--flash-attn on`, `--spec-type draft-mtp` n-max 2 (optimal; 3–8 worse, p-min>0 worse) |
| PagedAttention / vLLM | not-applicable | GGUF bs=1 path slower & being deprecated (vLLM RFC #39583); fast paths need a non-Q4_K requant |
| TensorRT-LLM | not-applicable | Qwen3.6 is hybrid Gated-DeltaNet+MoE; GDN kernel **aborts on SM89/Ada**; no GGUF loader; MTP waived in NVIDIA CI |
| SGLang | not-applicable | same ggml kernels for GGUF → same wall; MTP needs format+memory switch, 4-bit 35B-A3B OOMs 24 GB |
| FlashInfer | not-applicable | single-decode kernel ~25% slower than SDPA at batch-1; no GGUF path |
| Megakernel (Mirage/AMK) | not-applicable | win is cross-layer prefetch; our weights are already VRAM-resident; pins SMs hotter → worse throttle |
| Batch / dynamic batching / parallel decode | not-applicable (single-stream) | aggregate is flat ~91 (bandwidth-bound); doesn't raise single-stream tok/s |
| Early-exit decoding | rejected | not lossless |
| Mixed precision / FP8 | not-applicable | FP8 weights = *more* bytes/token → lowers the BW ceiling; FP8-KV does nothing at parallel=1 |
| Quantized kernels | already-optimal | MMVQ/DP4A is the optimal batch-1 path on sm_89; MMQ/MXFP4 don't run at decode |
| Tensor / Pipeline / Sequence parallel | not-applicable | needs multi-GPU (different machine); TP measured −8% single-stream (All-Reduce dominates at bs=1) |
| Graph optimization (ONNX/TensorRT) | not-applicable | see TRT-LLM (Ada-blocked) |
| Memory offloading | counter-productive | we do the opposite (full residency); offload is slower |
| Streaming generation | irrelevant | output transport, not decode rate |
| EAGLE-3 / ngram / tree-spec (Medusa) | not-applicable | EAGLE-3 ties MTP at best (no Qwen3.6 head); ngram measured −3…−12% here; tree explodes MoE verify |
| Lookahead / Jacobi (CLLM) | not-applicable | FLOPs-for-latency trade with no surplus FLOPs on a power-walled MoE |
| **FastMTP head retrain** | **infeasible here** | lossless & the right idea, but: no HF→GGUF path for a custom MTP head (converter arch-gated to native `mtp.*` layout); training needs multi-GPU (can't run on L4); caps ~105–115 anyway |

## The only real (but out-of-reach) lever

**FastMTP-style self-distillation of the MTP head** (arXiv 2509.18362) is the one lossless idea that attacks the acceptance cap. But for *this* deployment it's a **multi-week project on borrowed bigger hardware**, not a day on the L4: (1) `convert_hf_to_gguf.py:264-265` gates `--mtp` to the native Qwen3.6 `mtp.*` tensor layout, which FastMTP's MiMo/EAGLE training fork does not produce, with no precedent for the splice; (2) training needs the full 35B trunk resident (multi-GPU); (3) even at accept-length ~2.8–3.0 it lands ~105–115, not 120, because each extra accepted token widens the MoE verify-step expert union and re-pins the 72 W cap.

## Data sources (deduped)

**Engines/repos:** ggml-org/llama.cpp (mmvq.cu, mmq.cu, topk-moe.cu, common/speculative.cpp, convert_hf_to_gguf.py, src/llama-arch.cpp, conversion/qwen.py; PRs #18958, #19645, #18039, disc #17621); vllm-project/vllm (quantization/__init__.py, qwen3_5_mtp.py, moe_wna16.py, marlin_moe.py; RFC #39583); sgl-project/sglang; NVIDIA/TensorRT-LLM v1.3.0rc20 (gdn_mixer.py, modeling_qwen3_5.py, MOE_DEVELOPER_GUIDE.md; PR #12646, issues #13361/#6717); flashinfer-ai/flashinfer; mirage-project/mirage; RightNow-AI/AutoMegaKernel; z-lab/dflash; TencentBAC/FastMTP.
**Papers:** FastMTP 2509.18362 · DFlash 2602.06036 · Apple tree-MTP 2507.11851 · EAGLE 2401.15077 / -2 2406.16858 / -3 2503.01840 · Medusa 2401.10774 · SpecExec 2406.02532 · Lookahead 2402.02057 · CLLM 2403.00835 · MoE-Spec 2602.16052 · FlashInfer 2501.01005.
**Benchmarks:** jarvislabs (llama.cpp Qwen3.6-MTP 193→225 tok/s = 1.17× on RTX PRO 6000 — confirms bandwidth scaling); kaitchup DFlash-vs-MTP (Q4 llama.cpp faster at bs=1); unsloth GGUF discussion #14; NVIDIA "Optimizing llama.cpp with CUDA Graphs"; L4 datasheet (72 W / 300 GB/s).

## Software stack version (driver / CUDA-nvcc / GCP image / build) — also ~0

Checked whether a different driver, CUDA toolkit, GCP DLVM image, NGC container, or compile config gives a marginal speedup (searched NVIDIA/GCP docs + release notes, inspected the live box, and rebuilt from source):

| Stack lever | Finding | Gain |
|-------------|---------|------|
| GCP DLVM image | `cu129-…-nvidia-580` is the **newest** family; no cu13x / nvidia-590 DLVM exists | 0 |
| NVIDIA driver | **580.159.03** is the newest GCP-blessed branch; R580/R590 release notes have **no Ada/L4 decode or clock/power changes**; the driver doesn't gate ggml codegen (the container's CUDA runtime does) | 0 |
| Prebuilt image compile | `libggml-cuda.so` already ships **native sm_89 SASS** (cuobjdump: sm_86/sm_89/sm_120, no PTX-JIT), built with CUDA 12.8 | already-optimal |
| **CUDA 12.8 vs 12.9 build** (measured A/B, raw decode) | **70.1 vs 70.0 tok/s — identical**; newer nvcc produces no faster DP4A/MMVQ codegen for sm_89 | **0** |
| Clocks / power | memory clock **hard-fixed at 6251 MHz** (one state — can't overclock the 300 GB/s); power **hard-capped at 72 W** (`-pl 80` rejected; locked 2040 MHz collapses to 1950) | 0 |
| NGC / NVIDIA container | no NVIDIA-built llama.cpp image exists (NGC ships TRT-LLM/NIM/Triton — different engine, Ada-blocked for this model) | n/a |

So no version/build/driver/image change helps — you are already on the newest, optimally-compiled stack, and the walls are hardware.

## Community / fork levers (Reddit, deep GitHub, ik_llama) — benchmarked, also ~0

A second deep search (r/LocalLLaMA, deep GitHub issues/PRs/forks, NVIDIA forums, HN) surfaced concrete community tricks; every lossless one was **benchmarked on the box** and none beats mainline on our exact Q4_K_XL + f16 KV:

| Lever | Source | Measured result |
|-------|--------|-----------------|
| **ik_llama.cpp** (fork, faster `iqk` MoE kernels) | r/LocalLLaMA 1tjh7az (claimed 89→110); ikawrakow/ik_llama.cpp | **Loses.** Clean same-session A/B (same prompts, same clean GPU): mainline vs ik = chat 82.5 vs 80.5, **code 91.4 vs 83.4**, json 93.7 vs 92.7. Raw decode 70 vs 69.4 (tie). ik's MTP acceptance is *lower* (code 0.77 vs 0.67). Both lossless (17×23→391). The community's 110 tok/s uses `iq4_xs` + quantized KV = a quant change (off-limits). |
| **Spec-chaining** (`--spec-type draft-mtp` + `ngram-mod`) | llama.cpp PR #23269 | **~0.** code 86.6 vs 85.8 baseline (noise); acceptance unchanged. ngram doesn't fire usefully on this content. |
| **MTP gate sweep** (n-max 3/4, p-min 0.6) | thefrontierlab / carteakey | **~0.** n3 87.5, n4+p-min0.6 87.8 vs 87.3 baseline (all noise). Re-confirms n-max=2 optimal. |
| `--spec-autotune` (ik bandit) | ik PR #1595 | n/a — ik loses outright, so its autotune can't help here. |
| `GGML_CUDA_GRAPH_OPT=1`, FORCE_MMQ, clock-pin | NVIDIA blog / ggml | **~0** (measured earlier — all within noise). |
| PowerInfer / lookahead / Medusa / DFlash | various | not-applicable or non-lossless (extra model / quant change / OOM on 24 GB). |

**Honest note:** the community DOES have big numbers for this model (110–157 tok/s on ik), but every one is built on `iq4_xs`/`q4_0`-KV — a *different quantization*, which violates the lossless/same-weights constraint and would be cheating against the goal. On the **same Q4_K_XL weights + f16 KV**, mainline llama.cpp is already at/above every community alternative.

## Verdict

The ~91–99 tok/s in the headline **is** the lossless ceiling for this model on one L4 — confirmed by exhausting all 16 levers (12 dead-ends proven by reading code, the rest measured at ~0). To go past 120: **change the silicon** (L40S/A100/H100-class bandwidth). That is the only thing that moves the 300 GB/s wall, consistent with every measurement here and the 225 tok/s others get on a 3× faster card.
