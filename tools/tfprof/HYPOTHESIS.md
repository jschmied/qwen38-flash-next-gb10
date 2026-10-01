# Where a TensorFold Flash Next prompt's time goes (2026-10-01, before the profile)

§5an: prefill is nearly flat with length (EXL3 ~720 tok/s, our checkpoint ~1,300 tok/s bf16), so a per-token cost
dominates, not attention. One cold 8k prompt per checkpoint under nsys (`prefill_profile.py`, 0.6.0 stock).
- GPU busy ≥ 85 % of the prefill's wall time on both (it is GPU work, not Python or the n-gram gather).
- EXL3: the routed-expert prompt path (trellis decode + GEMM, `exl3` kernels) ≥ 50 % of GPU time; attention ≤ 10 %;
  the DeltaNet prompt chain ≤ 15 %.
- Our checkpoint: the NVFP4 expert prompt GEMMs ≥ 40 %; the block-FP8 / bf16 dense projections 15–30 %.
- If expert time dominates and scales with chunks (per-chunk weight decode), PREFILL_ROWS 4096/8192 should cut TTFT
  by ≥ 20 % on EXL3 (next test); if the GEMMs dominate at full tensor-core rate, chunk size won't help.

## Chunk-size sweep on EXL3 (after the profile: the grouped expert kernel is 56 % of an 8k prefill, all-expert work ~75 %)
Branch `prefill-rows-exp` (= ple-gather + `TF_PREFILL_ROWS` override, NOT for upstream), EXL3, one start per arm,
TTFT at 8k and 32k × 3 (`prefill_fast.json`), plus one greedy 64-token reply on the same 8k prompt per arm.
- If the grouped kernel decodes each expert's trellis once per chunk call (decode-bound, ~40 rows an expert at 2048),
  then 4096 rows: TTFT **−20…−35 %**; 8192: **−30…−45 %** vs 2048. If it is GEMM-bound per row, < 10 %.
- The greedy reply is identical in all three arms (TensorFold's prompt passes are split-invariant); a difference
  means chunk size changes bits.
- 8192 may fail to start (buffers sized elsewhere assume 2048): a start failure is a result, not a reason to stop.

## T9 quick test (shortest-first fill), EXL3, small mix (1 × 32k prompt at t=0, 4 × 2k every 10 s)
Stock 0.6.0 vs branch `ple-gather` (default chunk rows), one start each; `longmix.py <tag> <tok> 1 32000 4 2000 10`;
then `bench_concurrent --alone --serial --levels 1,2` on the branch.
- Stock: short requests' TTFT ≈ the time left on the 32k prompt (~45 s at ~720 tok/s, minus their arrival offset).
- Branch: short requests' TTFT **2.5–8 s** (their own ~2.8 s prefill plus at most one other pass); the long prompt's
  TTFT grows by ≤ 4 × ~3 s.
- Exactness on the branch: all alone/serial hashes equal.
