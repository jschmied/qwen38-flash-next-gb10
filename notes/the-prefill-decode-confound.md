# ~~Ranked on prefill, measured on decode~~ — REFUTED by a decode-only profile

> **This note's central claim is wrong and is kept for the record.** A decode-only re-profile
> (12-token prompt, 600 generated tokens, warmed) produced essentially the same ranking as the
> mixed one: FP8 blockwise 34.5% (was 32.2%), BF16 wmma 26.6% (was 25.0%), MoE grouped 22.2%
> (was 22.1%). The hyper-connections really are ~27% of *decode* GPU time. The prefill
> contamination was a plausible mechanism that testing did not support.
>
> The three nulls remain unexplained. What is ruled out: mis-ranked layers, an idle GPU
> (89.8% busy on the decode trace), and non-dispatching kernels (verified at the call site).
> Live hypothesis: FP8 does not speed up *these particular shapes* —
> `(10240, 320)` at M<=8 through `scaled_mm`, where K=320 is small enough that halving bytes may
> buy nothing against a warp-level BF16 WMMA that is already latency-bound. Measurable per shape
> offline; not yet measured.


## What happened

We profiled the running model, found **25% of GPU kernel time in
`cutlass_80_wmma_tensorop_bf16`**, identified those kernels as the hyper-connections, and spent a
day attacking them. Three independent interventions, all measuring nothing:

| intervention | what it changed | result |
| --- | --- | --- |
| blockwise FP8 (`FP8_PB_WO`) | precision | **slower** — 128-divisibility fails, falls to Triton |
| CUTE-DSL skinny GEMM (gate + M=3) | kernel for the fused down projection | 35.92 vs 36.45, null |
| per-channel FP8 up projection | half the bytes **and** the faster kernel | 36.05 vs 36.45, null |

## The cause

**The profile was prefill-dominated; the benchmark is decode-only.**

That trace was captured over `bench_client_real.py --input-len 4000 --requests 4`:

```
prefill tokens   4 x 4000 = 16,000
decode tokens    4 x  512 =  2,048     -> 8:1 in favour of prefill
```

The dispatch census proves it directly — the same layers are called at
**M = 832, 2552, 3224, 4096** (prefill batches) as well as M = 1, 2, 4, 8 (decode). A GEMM at
M=4096 is a different machine-level problem from the same GEMM at M=1: large, tiled, efficient,
and nothing like the skinny shape decode issues.

`decode_tps_median` — what every arm was scored on — **excludes prefill entirely**. So targets
were ranked on one workload and the fix measured on another. Any layer whose share is large in
prefill and small in decode looks like a lever and behaves like a placebo.

## What it is not

Checked before blaming the profile, because "the GPU is idle so kernels don't matter" would have
been the other explanation:

```
trace span         3.718 s
GPU busy (union)   3.177 s   = 85.4%
idle in gaps       0.541 s   = 14.6%   (77,685 gaps; only 375 exceed 100us)
```

The GPU is genuinely working. Kernel time *is* wall time here. The ranking was wrong, not the
premise that kernel time matters.

## The rule

**Profile the regime you intend to measure.** A decode-truthful profile needs a short prompt and a
long generation (we use ~12 prompt tokens, 600 generated, `ignore_eos`), warmed first so JIT and
cudagraph capture stay out of the trace. A profile taken over a realistic serving mix answers
"where does this workload spend time", which is a different question from "what should I optimise
to raise decode tok/s".

Corollary for anyone reading our published numbers: the per-token **byte** budget in
`bench/flashnext-quants` is decode-specific and correct; the **25% GPU-time** attribution that sat
next to it was not, and the two looked consistent only by accident.
