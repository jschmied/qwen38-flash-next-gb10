# Switching to a dense-quantized checkpoint: +71% single-stream

`lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` quantizes the attention and GDN projections that
[the profile](single-stream-limit.md) indicts, and keeps the compact FP8 PLE. It is strictly
better than `RadixArk/Qwen3.8-Flash-Next-NVFP4` on every axis we measured — faster, smaller, and
with no measurable quality cost.

Measured 2026-08-28, one GB10, bare metal, `VLLM_PLE_CPU_OFFLOAD=1`, 8192 ctx.

## Results

| c | RadixArk | FP8-mixed | FP8-mixed + MTP k=2 | best vs baseline |
|---:|---:|---:|---:|---:|
| 1 | 17.1 | 23.7 | **29.2** | **1.71x** |
| 2 | 33.4 | 42.2 | **51.1** | 1.53x |
| 4 | 44.1 | 68.3 | **69.0** | 1.56x |
| 8 | 87.5 | **105.5** | 98.8 | 1.21x |
| 16 | 131.6 | **156.0** | 139.6 | 1.19x |

Also **smaller**: 74.13 GiB resident vs 76.61, which buys 2.5 GiB more KV (33.47 vs 30.99).
TTFT improves too — 0.19 s at c=1 against 0.22.

**Quality: no measurable cost.** NLL/token on identical held-out chunks:

| | RadixArk | FP8-mixed |
|---|---:|---:|
| NLL/token | 0.7748 | 0.7610 |
| perplexity | 2.17 | 2.14 |

Five of six chunks improved, aggregate −1.8% relative. We report this as **no regression**, not as
an improvement: 276 tokens is a small sample, and "quantization improved quality" is not a
mechanism we would defend. It does settle the question that mattered — lovedheart's published
GSM8K/AIME files describe a *different* build (see [failure-modes.md](failure-modes.md) D4b), so
this was the only quality evidence available.

The bandwidth model predicted +32% at c=1; we measured +39% (and +71% with MTP). Close enough to
confirm the mechanism.

## The two optimizations are substitutes, not complements

MTP was worth **+67%** on RadixArk. On this checkpoint it is worth **+23%**, and above c=4 it is a
**net loss**:

| | c=1 | c=8 |
|---|---|---|
| MTP on RadixArk | 17.1 → 28.5 (**+67%**) | 87.5 → 89.0 (+2%) |
| MTP on FP8-mixed | 23.7 → 29.2 (**+23%**) | 105.5 → 98.8 (**−6%**) |

Both levers attack the same cost — the per-token weight read. Speculation amortizes it across
drafted tokens; the checkpoint shrinks it. Having shrunk the read by ~25%, there is proportionally
less for speculation to recover, while its compute cost for drafting and verification is
unchanged. Past c=4 that cost exceeds the saving.

**So the right configuration depends on load**, and stacking every optimization is wrong:

| workload | configuration | result |
|---|---|---|
| single user | FP8-mixed **+ MTP k=2** | 29.2 tok/s |
| c ≤ 4 | FP8-mixed **+ MTP k=2** | 51–69 tok/s |
| c ≥ 8 | FP8-mixed, **speculation off** | 105–156 tok/s |

Acceptance on this checkpoint: 68.4% / 46.7% at positions 0–1, mean accepted length 2.15 —
essentially unchanged from RadixArk's 2.16 under load, so the shrinking benefit is *not* the
drafter getting worse. It is the target getting cheaper.

## Getting it to load: six defects

Five were silent or misleading. Full detail in [failure-modes.md](failure-modes.md); the short
version, in the order they appear:

| # | defect | presents as |
|---|---|---|
| 1 | `FP8_PB_WO` undispatched (stale image) | **silent garbage** |
| 2 | PLE gate rejects `modelopt_mixed` | clean error |
| 3 | scale named `weight_scale_inv` | misleading `AttributeError` (vllm#53107) |
| 4 | scale rank 2-D vs 4-D | bare `assert`, no message |
| 5 | tensor subclass leaks into `Parameter()` | our own regression from fixing #4 |
| 6 | **DeepGEMM faults on sm_121** | `CUDA error: unspecified launch failure` |

**#3 and #4 are one root cause**: two ModelOpt `FP8_PB_WO` export conventions exist. vLLM
implements `weight_scale` rank-4 (per the closed vllm#30938, "4D block scales"); ModelOpt 0.46.0
also emits `weight_scale_inv` rank-2 — the DeepSeek convention that `fp8.py` already implements
elsewhere. `_inv` is a naming legacy, **not a reciprocal**; we verified that before bridging them,
because a reciprocal would have produced plausible-but-wrong output rather than a crash.

**#6 will hit anyone else doing this.** `VLLM_USE_DEEP_GEMM=0` is required on GB10:

```
torch.AcceleratorError: CUDA error: unspecified launch failure
  vllm/utils/deep_gemm.py:464 in fp8_gemm_nt
```

DeepGEMM's support gate is `is_device_capability_family(100) or is_device_capability_family(120)`.
GB10 is **sm_121**, which satisfies the family-120 check — so vLLM routes blockwise FP8 to a
kernel that then faults. Disabling it falls back to `CutlassFp8BlockScaledMMKernel`, which works.
Same shape as the trtllm-gen SM100-only bug: a capability-*family* check too coarse for GB10. It
is reachable only through the blockwise-FP8 path, which is why nothing on this box exercised it
until the dense weights were quantized.

Also required with FP8 GDN projections: `VLLM_GDN_DECODE_KERNEL=triton`, or the default CUDA
kernel hangs the engine at concurrency ~32 with no error.

## Positioning

29.2 tok/s single-stream still trails the field's best published free-form figure
(34.8, Death-By-Tokens, SGLang + HashK-compressed PLE + NEXTN k=3), but the gap is now small.
On aggregate, 156 tok/s at c=16 with speculation off.

The remaining dense weight is 1.64 B / 3.28 GB per token — `lm_head`, the hyper-connection blocks
and `shared_expert`. See [single-stream-limit.md](single-stream-limit.md) for what quantizing
those is worth and which are safe.
