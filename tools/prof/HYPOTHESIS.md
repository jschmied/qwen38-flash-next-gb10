# Whole-step kernel profile, c=1 MTP n=3 decode (speed-of-light step 3), written 2026-09-25 00:3x before the run

Context: every dense module (FP8 projections, BF16 mixer/shared/router linears, lm_head, target MoE) measured at
0.94-1.16x its byte floor in isolation (steps 2a-2c), yet the verify cycle is 59.2 ms against a 45.2 ms floor
(14 ms gap). Prod config: PIECEWISE compiled, NOT --enforce-eager.

Expected:
- GPU busy time per cycle 50-59 ms (little idle; FULL decode graphs were null, finding 237), idle 0-9 ms.
- The dense-module categories (FP8 GEMM, BF16 GEMM, MoE grouped GEMM, lm_head) sum to ~40-45 ms per cycle.
- The remainder, ~10-15 ms, sits in categories the module benches never saw: GDN recurrence/conv/gating (36 layers),
  QSA indexer/top-k/attention (12 layers), norms/RoPE/activation-quant/elementwise, PLE, sampling/rejection, the
  drafter's non-dense work.
- Outside: if idle > 10 ms per cycle, the gap is host-side after all (contradicting 237 -> check that FULL really
  replayed); if the dense categories sum to much more than the module benches predict, in-model conditions (L2
  contention, concurrent streams) differ from the isolated bench.
