## Purpose

On SM 12.x (GB10 / DGX Spark, RTX PRO 6000 Blackwell) the CUTLASS blockwise-FP8 GEMM (`CutlassFp8BlockScaledMMKernel`, 128×128 weight blocks, 1×128 activation groups) loses most of its throughput once M spans many tile rows. Measured on a GB10 (48 SMs, 24 MiB L2) for a 16384×2560 projection, median of 5×10 launches:

| M (rows) | single launch | 4096-row chunks |
| --- | --- | --- |
| 4,096 | 163 TFLOPS | — |
| 8,192 | 95 TFLOPS (7.6 ms) | **161 TFLOPS (4.3 ms)** |
| 16,384 | 53 TFLOPS (26.2 ms) | **160 TFLOPS (8.6 ms)** |
| 32,768 | 52 TFLOPS (52.9 ms) | **161 TFLOPS (17.0 ms)** |

cuBLASLt's row-wise FP8 path degrades identically (99 → 52 → 53 TFLOPS), its per-tensor path does not (~175 TFLOPS at every M), which is consistent with the weight operand (42 MB here) no longer staying L2-resident across tile rows. Any blockwise-FP8 checkpoint served on these GPUs with `--max-num-batched-tokens` ≥ 8192 pays 1.7–3× on every FP8 projection during prefill (on Qwen3.8-Flash-Next with FP8 projections these GEMMs are ~half of prefill time).

## Changes

- `CutlassFp8BlockScaledMMKernel.apply_block_scaled_mm`: on SM 12.x, issue the GEMM in 4,096-row chunks (`chunked_blockwise_scaled_mm`). Each chunk's activation scales are re-materialised in the kernel's column-major layout — the kernel derives that layout from its own M, so a row slice of the full scale tensor is read with the wrong stride (measured: silently wrong results).
- Other architectures unchanged (`blockwise_fp8_m_chunk()` returns 0).

The chunked result is bit-identical to the single launch (the per-element K-reduction is unchanged); verified against an fp32 reference at FP8 quantisation noise.

## Test plan

- New `test_cutlass_fp8_blockwise_m_chunk_matches_single_launch` in `tests/kernels/quantization/test_cutlass_scaled_mm.py`: M ∈ {8192, 12288} × (N,K) ∈ {(2048,2560), (2560,6144)} × chunk ∈ {4096, 2048}, `torch.equal` against the single launch, with column-major scales as `QuantFP8(column_major_scales=True)` emits them. Runs on any SM90+ GPU (the chunking function is architecture-independent; only the dispatch is gated).
- The eight parametrisations were run on a GB10 (SM 12.1) through the same function: 8/8 bit-identical.

## Test result

GB10: microbenchmark above; new test 8/8. Not measured on SM 12.0 (RTX PRO 6000) — same L2 class, expected to behave the same; the gate is the capability family, please say if it should be narrower.

A tile-scheduler raster/swizzle change inside the kernel would be the in-kernel version of the same fix; this PR takes the Python route because it is exact, small, and reversible.

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
