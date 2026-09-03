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

- `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`: when M > 4,096 **and the FP8 weight operand (N×K bytes) exceeds the device L2** (`cudaDeviceProp::l2CacheSize`), issue the GEMM in balanced launches of at most 4,096 rows (the fewest chunks, equal sizes, starts kept multiples of 4 — M=4097 becomes two ~2,048-row launches, never 4096 + 1). A and the output are row-range views written in place; each chunk's activation scales are re-laid out in the kernel's column-major layout (the kernel derives that layout from its own M, so a row slice of the full scale tensor is read with the wrong stride — measured: silently wrong results). No extra output buffer, no Python-side control flow: the model graph still sees one `cutlass_scaled_mm` op with symbolic M.
- The gate is the actual condition, not the architecture family: GB10 (24 MiB L2) chunks a 42 MB weight; RTX PRO 6000 Blackwell / GB202 (96–128 MiB L2) do not, and nothing else changes for them. SM 9.x/10.x paths are untouched.
- The chunked result is bit-identical to the single launch (the per-element K-reduction is unchanged); verified against an fp32 reference at FP8 quantisation noise.

Review history on this PR: the first revision did the loop in Python (`apply_block_scaled_mm`) and gated on `is_device_capability_family(120)`; both were changed after review — shape-dependent Python control flow in the compiled hot path specialises Dynamo on the token count, and the family gate covered parts whose L2 holds the weight.

## Test plan

- `test_cutlass_fp8_blockwise_large_m` (M ∈ {4096, 4097, 8193, 12288} × (N,K) ∈ {(2048,2560), (5120,5120), (16384,2560), (57344,2560)}): the 25 MiB, 42 MB and 147 MB weights exceed the 24 MiB (GB10) and 128 MiB (GB202) L2 respectively, so the chunk path is exercised on every SM 12.x part; 5 MB is the no-chunk control. Every case is checked against the dequantized fp32 baseline; on SM 12.x the op is additionally asserted **bit-identical** to a Python reference doing the same balanced chunking. On other architectures only `assert_close` is used, since e.g. the SM90 dispatch may pick a different kernel configuration per launch (swap-AB on M % 4) and bit-equality across launches is not a contract there.
- `test_cutlass_fp8_blockwise_compiled_dynamic_m`: one `torch.compile(dynamic=True, fullgraph=True)` graph on the 42 MB weight serving M = 4096, 4097, 8193, 12288, equal to eager, with `CompileCounterWithBackend` asserting `frame_count == 1` — the M dispatch is inside the op, no recompilation on the token count.
- Standalone build of the modified `.cu` against CUTLASS v4.7.1 on a GB10 (SM 12.1): bit-identity against the unmodified op and a throughput sweep around the threshold (M = 64, 4096, 4097, 5120, 6144, 8191, 8192, 8193, 12288, 16384 for the 42 MB, 147 MB and 5 MB weights) — **pending, will be posted here** (draft until then).

## Test result

Microbenchmark evidence above is from the same kernel launched in chunks from Python on a GB10 (finding-level numbers, 5×10 launches, median). The C++ path's own numbers follow once the standalone build has run. Not measured on SM 12.0 parts; by construction they are not chunked unless the weight exceeds their L2.

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
