# Why 27% of decode time did not respond to anything

Three interventions on the hyper-connections all measured null. The layers were correctly
ranked and the GPU was busy, so the nulls needed an explanation. Here it is, measured per shape
(`tools/shapebench.py`).

## The measurement

FP8 `_scaled_mm` against BF16 `F.linear`, at the shapes and batch sizes decode actually uses.
L2 defeated by rotating over ~300 MB of distinct weights; every timing printed next to its
roofline (`N*K*2 / 273 GB/s`) so a cache-resident lie is visible rather than believed.

| shape | role | M=1 BF16 | M=1 FP8 | speedup | roofline |
| --- | --- | --- | --- | --- | --- |
| (10240, 320) | hyper-connection **up** | 30.8 us | 26.1 us | **1.18x** | 24.0 us |
| (336, 10240) | hyper-connection **down** | 37.0 us | 41.5 us | **0.89x** | 25.2 us |
| (10240, 2560) | GDN in_proj (control) | 307 us | 126.7 us | **2.42x** | 192 us |

Flat across M = 1, 2, 4, 8 in every case.

## The answer

**These GEMMs are latency-bound, not bandwidth-bound.** At 30–37 us against a 24–25 us
roofline they already run at ~78% of peak for their size, and they do not move with precision.
The control shape — where FP8 genuinely helps — has room (307 us against a 192 us roofline) and
delivers 1.8–2.4x.

They are 27% of decode GPU time because there are **~102,000 of them**, not because any one is
expensive: roughly 200 per forward (48 layers x 2 hyper-connections x {down, up}), ~39 us each
in the live trace, which matches the microbenchmark.

**And the two halves cancel.** FP8 saves ~4 us/call on `up` and *loses* 6–9 us/call on `down`.
Our up-only variant should have bought ~1.3% of GPU time; we measured −1.1% ± 3.0%. The nulls
were the correct answer, not a failed measurement.

## What this closes, and what it opens

Closed: precision and kernel choice on these layers. Blockwise FP8, per-channel FP8, and the
CUTE-DSL skinny GEMM are all attacking a bandwidth problem that does not exist here. Do not
retry them, and do not read the 27% as a bandwidth opportunity.

Open, and the only remaining lever: **the launch count.** ~200 small GEMMs per forward at a
~30 us floor is ~6 ms per forward that is structural. Fixing it means fusing or eliminating
launches, not making each one cheaper — e.g. batching the two hyper-connections within a layer,
or a fused kernel for the whole mHC residual (b12x ships one, `norm.mhc`, but it is
DeepSeek-style and restricted to hidden sizes 4096/7168; ours is 2560, so it does not apply).

## Method note

This microbenchmark should have come first. It costs about two minutes with the server stopped
and it would have pre-empted a checkpoint build, four failed server starts, and three six-run
A/B arms. Rank by profile, then **verify the lever is real at the shape level before building
anything.**
