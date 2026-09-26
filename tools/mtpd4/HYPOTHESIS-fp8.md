A2b FNMTPDENSE8, written 2026-09-26 before the run (user: "fp8 drafter then instead of fp16").
Context: A2 NVFP4 (mtpd4a, 1 start per arm, stopped early): accept_len 2.443 vs 2.53, ms/tok 21.725 vs 21.237 (+2.3 %)
-> the drafter's dense layers are precision-sensitive at NVFP4 (rel. weight error ~9.5 %).
Change: same four layers, FP8 E4M3 with one FP32 scale per output row (rel. error ~2.6 %), Triton rows GEMV.
Microbench: 698 -> 311 us per M=1 draft step, 568 -> 308 us at M=4 -> ~-1.03 ms per verify cycle.
H1: accept_len within -0.03 of base; per cycle -0.7 ... -1.1 ms; ms/tok -1.3 ... -2.0 %.
H0: accept_len drops > 0.05 -> even FP8 per-row hurts; then per-layer probe (FN_MTP_DENSE_LAYERS) to find the sensitive one.
