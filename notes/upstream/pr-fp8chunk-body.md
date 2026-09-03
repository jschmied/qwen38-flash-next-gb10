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

- `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`: when M > 4,096 **and the FP8 weight operand (N×K bytes) exceeds the device L2** (`cudaDeviceProp::l2CacheSize`), issue the GEMM in 4,096-row launches. A and the output are row-range views written in place; each chunk's activation scales are re-laid out in the kernel's column-major layout (the kernel derives that layout from its own M, so a row slice of the full scale tensor is read with the wrong stride — measured: silently wrong results). No extra output buffer, no Python-side control flow: the model graph still sees one `cutlass_scaled_mm` op with symbolic M.
- The gate is the actual condition, not the architecture family: GB10 (24 MiB L2) chunks a 42 MB weight; RTX PRO 6000 Blackwell / GB202 (96–128 MiB L2) do not, and nothing else changes for them. SM 9.x/10.x paths are untouched.
- The chunked result is bit-identical to the single launch (the per-element K-reduction is unchanged); verified against an fp32 reference at FP8 quantisation noise.

Review history on this PR: the first revision did the loop in Python (`apply_block_scaled_mm`) and gated on `is_device_capability_family(120)`; both were changed after review — shape-dependent Python control flow in the compiled hot path specialises Dynamo on the token count, and the family gate covered parts whose L2 holds the weight.

## Test plan

- `test_cutlass_fp8_blockwise_large_m` (M ∈ {4096, 8193, 12288} × (N,K) ∈ {(2048,2560), (16384,2560), (57344,2560)} — the 42 MB and 147 MB weights exceed the 24 MiB (GB10) and 128 MiB (GB202) L2 respectively, so the chunk path is exercised on every SM 12.x part; the 5 MB weight is the no-chunk control): `torch.equal` against a Python row-chunked reference with per-chunk re-laid-out scales (what the C++ path does internally; on GPUs that do not chunk it is the same identity), plus the usual tolerance check against the dequantized fp32 baseline.
- `test_cutlass_fp8_blockwise_compiled_dynamic_m`: one `torch.compile(dynamic=True, fullgraph=True)` graph serving M = 4096, 8193 and 12288 on the 42 MB weight, equal to eager — the M dispatch is inside the op.
- Standalone build of the modified `.cu` against CUTLASS v4.7.1 on a GB10 (SM 12.1): bit-identity against the unmodified op at every tested (M, N, K) incl. partial last chunks, plus throughput — **pending, will be posted here** (draft until then).

## Test result

Microbenchmark evidence above is from the same kernel launched in chunks from Python on a GB10 (finding-level numbers, 5×10 launches, median). The C++ path's own numbers follow once the standalone build has run. Not measured on SM 12.0 parts; by construction they are not chunked unless the weight exceeds their L2.

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
