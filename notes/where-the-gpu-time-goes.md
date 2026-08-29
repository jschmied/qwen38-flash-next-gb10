# Where the GPU time actually goes (profiled, not inferred)

Two experiments — fused draft metadata, full CUDA graphs — were run on hypotheses reasoned from
timing arithmetic. Both were null. This is what a profile says instead.

Config: `fp8head`, MTP k=2, PIECEWISE, single stream. Trace summary in
`data/profile-fp8head-mtp2-kernels.txt`. **3,602 ms of GPU kernel time across 145,824 launches.**

| kernel | ms | % | calls | us/call |
| --- | --- | --- | --- | --- |
| `cutlass_3x_gemm_fp8_blockwise` | 1160 | 32.2% | 5643 | 206 |
| GroupedGEMM (NVFP4 MoE) | 532 | 14.8% | 2640 | 201 |
| **`cutlass_80_wmma_tensorop_bf16`** | 467 | 13.0% | 12307 | 38 |
| **`cutlass_80_wmma_tensorop_bf16`** | 432 | 12.0% | 10432 | 41 |
| GroupedGEMM | 264 | 7.3% | 2640 | 100 |
| **`fused_moe::Fused_Moe_Kernel_sm80`** | 58 | 1.6% | 55 | **1058** |
| `MoeFCGemm` | 27 | 0.7% | 55 | 488 |

## The hyper-connections are 25% of all GPU time

The two `cutlass_80_wmma_tensorop_bf16` entries total **900 ms — 25%** of GPU time across ~22,700
launches. Launch geometry identifies them: `block=(32,1,1)` is a **single warp**, and the call
counts (5,656 and 5,600) match **2 hyper-connections x 48 layers x ~59 forwards = 5,664**.

`sm80` is Ampere. These are legacy warp-level WMMA kernels running on an SM121 Blackwell part.

**This explains the failed `_up` experiment.** Quantizing the hyper-connection `_up` projection to
FP8 measured *slower* (35.0 vs 36.3 tok/s) and we recorded it as "a lever we measured and
rejected". The precision was never the problem: we moved those layers off a fast warp-WMMA BF16
kernel onto a slower Triton blockwise FP8 GEMM. The lever is the **kernel**, not the bits — and
the 128-divisibility constant that blocks the fused `[320,4,12]` down-projection is a separate
issue from this one.

## The drafter's MoE runs an Ampere unquantized path at 5x the cost

`Fused_Moe_Kernel_sm80` (1058 us/call) and `MoeFCGemm` (488 us/call) appear ~57 times each —
matching the draft steps (2 per target forward). Together ~1.8 ms per draft step against a ~10 ms
draft step, so **~18%**.

The quantized body MoE runs GroupedGEMM at **201 us/call**. The drafter's is 5x slower per call
because `mtp.layers.0.mlp.experts.*` is unquantized BF16 (`ignore` contains `mtp.*`), forcing the
sm80 unquantized MoE path.

So quantizing the drafter's experts is worth ~16% of a draft step (~5% of an iteration) — **not**
via bytes, as previously claimed here, but via kernel selection. It also lifts the global
`--moe-backend` veto (see `moe-backend-axis.md`).

## What this corrects

- Earlier claim: "the drafter's 4.86 GiB unquantized MoE is what caps speculation." Wrong on the
  mechanism and the size. The experts are only 0.095 GiB read per draft token (10 of 512); the
  cost is the **sm80 kernel**, worth ~5% of an iteration, not a headline lever.
- Earlier claim: hyper-connection quantization is "a lever we measured and rejected." Accurate as
  a result, wrong as a conclusion — it was rejected because the replacement kernel was worse.

## Ranked by measured GPU time

1. **FP8 blockwise GEMM, 32%** — our own dense quantization, already the fast path.
2. **NVFP4 MoE GroupedGEMM, 22%** (532 + 264) — already fast.
3. **Hyper-connections in sm80 BF16 WMMA, 25%** — untouched, and the largest addressable item.
4. Drafter MoE on the sm80 unquantized path, ~2%.
