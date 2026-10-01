# Where a TensorFold Flash Next prompt's time goes (2026-10-01, before the profile)

§5an: prefill is nearly flat with length (EXL3 ~720 tok/s, our checkpoint ~1,300 tok/s bf16), so a per-token cost
dominates, not attention. One cold 8k prompt per checkpoint under nsys (`prefill_profile.py`, 0.6.0 stock).
- GPU busy ≥ 85 % of the prefill's wall time on both (it is GPU work, not Python or the n-gram gather).
- EXL3: the routed-expert prompt path (trellis decode + GEMM, `exl3` kernels) ≥ 50 % of GPU time; attention ≤ 10 %;
  the DeltaNet prompt chain ≤ 15 %.
- Our checkpoint: the NVFP4 expert prompt GEMMs ≥ 40 %; the block-FP8 / bf16 dense projections 15–30 %.
- If expert time dominates and scales with chunks (per-chunk weight decode), PREFILL_ROWS 4096/8192 should cut TTFT
  by ≥ 20 % on EXL3 (next test); if the GEMMs dominate at full tensor-core rate, chunk size won't help.
