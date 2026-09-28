# MoE prefill fusion in Triton (FUSION-AGENDA item 3, TODO item 6) — hypothesis before any timing

Written 2026-09-28, before the first timed run. User: "go for 3".

## What changed the route

- The ledger trace (`/opt/llm/capture/ledger/fn.nsys-rep`, one 3,456-token chunk, one MoE layer) splits the MoE into
  expand + fp4 quant + prefix sums 0.57 ms, **GEMM1 4.18 ms** (CUTLASS SM120 `128x64x256`, persistent, 48 CTAs),
  `doActivation` 0.50 ms, GEMM2 2.71 ms, finalize 0.86 ms: **8.9 ms per layer-chunk**.
- GEMM1 moves ~1.02 GB (839 MB of expert weights, 52 MB of their scales, 44 MB expanded A, 88 MB bf16 D) in 4.18 ms
  (≈ 245 GB/s, the DRAM floor) while doing 226 GFLOP (≈ 54 TFLOPS, ~11 % of the FP4 MMA peak). A fused kernel only has
  to **stream the weights at the floor**; its MMA efficiency hardly matters.
- Triton 3.7.1 lowers `tl.dot_scaled(e2m1, e4m3 scales)` to the native `mma…kind::mxf4nvf4.block_scale.scale_vec::4X`
  when compiled for sm_120 (`TRITON_OVERRIDE_ARCH=sm120`; the sm_121 gate is Triton #10010, not in 3.7.1). Exact
  against a dequantized reference (`dsprobe2.py`, max abs err 0). So the fusion is a Triton kernel with minute-scale
  iterations, not a CUTLASS port with 12-minute rebuilds.

## Design (prefill only, above a token threshold; decode keeps FlashInfer)

1. Quantize x once per token to NVFP4 (FlashInfer's fast-math recipe, `a1_gscale`), linear scales.
2. `moe_align_block_size` → sorted (token, k) rows per expert block.
3. **GEMM1 + SwiGLU + FP4**: A rows gathered from the per-token tensor (no `expandInputRows`; the 4.4 MB per-token A
   stays in L2), gate and up tiles in the same CTA (w13 is `[up | gate]` after vLLM's w1w3→w3w1 reorder), epilogue:
   `alpha_e`, bf16 round, `silu(gate)·up`, bf16 round, FP4 quant with `a2_gscale` → intermediate FP4 + linear scales.
   Removes `doActivation` and GEMM1's 88 MB bf16 write.
4. GEMM2 (Triton, same weights in place, swizzled scales read by index math), `alpha2`, bf16 rows per (token, k).
5. Fixed-order finalize (k = 0..9 sequential, fp32, router weights), deterministic like prod's DETFIN.

No weight copies: all kernels read FlashInfer's processed tensors (a second copy would be ~5 GB).

## Predictions (ranges)

| stage | prediction | kill criterion |
|---|---|---|
| S1 GEMM1+act+quant standalone, M = 3,456, random routing | **3.9–4.6 ms** (floor ≈ 3.9 ms) | > 4.8 ms: the chain cannot beat GEMM1 + `doActivation` (4.68 ms); stop |
| S1 correctness | FP4 codes identical to a reference built from the same fp32 math on ≥ 99 % of elements, the rest one code step (accumulation order differs); scales identical on ≥ 99.5 % | a code more than one step off, or any NaN |
| S2 GEMM2 standalone | 2.6–3.2 ms | > 3.3 ms |
| S3 whole prefill MoE vs FlashInfer, same process, same inputs | −0.7…−1.3 ms of 8.9 ms per layer-chunk; output rel L2 ≤ 1e-2 vs FlashInfer; bit-identical run to run | slower, or rel L2 > 3e-2 |
| S4 server A/B (2 starts per arm, prod config, clone venv) | TTFT −2.5…−5 %; decode unchanged | |
