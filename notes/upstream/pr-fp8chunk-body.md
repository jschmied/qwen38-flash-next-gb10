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

- `csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm120_fp8.cu`: when the FP8 weight operand (N×K bytes) exceeds the device L2 (`cudaDeviceProp::l2CacheSize`), issue the GEMM in balanced row chunks sized to keep ~12 MiB of activation per launch (at most 4,096 rows, at least 512, 4-aligned), starting once M is ≥ 1.5 chunks. A and the output are row-range views written in place; each chunk's activation scales are re-laid out in the kernel's column-major layout (the kernel derives that layout from its own M; a row slice of the full scale tensor is read with the wrong stride — measured, silently wrong). No extra output buffer, no Python-side control flow: the model graph sees one `cutlass_scaled_mm` op with symbolic M.
- The gate is the actual condition, not the architecture family: GB10 (24 MiB L2) chunks a 42 MB weight; RTX PRO 6000 Blackwell / GB202 (96–128 MiB L2) do not, and nothing changes for them or for SM 9.x/10.x.
- The chunked result is bit-identical to the single launch (per-element K-reduction unchanged).

## Measurements (GB10, same build flags and CUTLASS v4.7.1 for both columns, 5×10 launches, median)

| weight (N×K, FP8) | M | unmodified | this PR | ratio |
| --- | --- | --- | --- | --- |
| 16384×2560 (42 MB) | 4096 | 2.14 ms (161 TF) | 2.14 ms | 1.00 |
| | 4097 | 2.12 ms | 2.11 ms | 1.01 |
| | 6144 | 3.44 ms (150 TF) | 3.37 ms | 1.02 |
| | 8192 | 7.24 ms (95 TF) | **4.38 ms (157 TF)** | 1.66 |
| | 12288 | 19.5 ms (53 TF) | **6.59 ms (156 TF)** | 2.96 |
| | 32768 | 51.4 ms (54 TF) | **17.7 ms (155 TF)** | 2.90 |
| 57344×2560 (147 MB) | 8192 | 25.8 ms (93 TF) | **15.6 ms (155 TF)** | 1.66 |
| | 16384 | 94.9 ms (51 TF) | **30.0 ms (160 TF)** | 3.16 |
| 5120×5120 (25 MiB) | 4096 | 1.90 ms (113 TF) | **1.35 ms (159 TF)** | 1.41 |
| | 8192 | 5.74 ms (75 TF) | **2.70 ms (159 TF)** | 2.12 |
| | 32768 | 22.4 ms (77 TF) | **10.8 ms (159 TF)** | 2.08 |
| 2048×2560 (5 MB, not chunked) | 4096…32768 | — | — | 0.98–1.01 |

48 (M, N, K) points incl. 64, 2048, 4097, 5120, 8191, 8193: **48/48 bit-identical**, no point slower than 0.98×. Full table: [fp8chunk_standalone_v2.txt](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/data/fp8chunk_standalone_v2.txt). Earlier context: the same collapse was first seen end to end as a 4.6 s vs 3.1 s TTFT difference between batch 8192 and 4096 on Qwen3.8-Flash-Next with FP8 projections (these GEMMs are ~half of its prefill).

## Test plan

- `test_cutlass_fp8_blockwise_large_m` (M ∈ {4096, 4097, 8193, 12288} × (N,K) ∈ {(2048,2560), (5120,5120), (16384,2560), (57344,2560)}): the 25 MiB, 42 MB and 147 MB weights exceed the 24 MiB (GB10) and 128 MiB (GB202) L2 respectively, so the chunk path is exercised on every SM 12.x part; 5 MB is the no-chunk control. Every case is checked against the dequantized fp32 baseline; on SM 12.x the op is additionally asserted **bit-identical** to a Python reference doing the same balanced chunking. On other architectures only `assert_close` is used, since e.g. the SM90 dispatch may pick a different kernel configuration per launch (swap-AB on M % 4) and bit-equality across launches is not a contract there.
- `test_cutlass_fp8_blockwise_compiled_dynamic_m`: one `torch.compile(dynamic=True, fullgraph=True)` graph on the 42 MB weight serving M = 4096, 4097, 8193, 12288, equal to eager, with `CompileCounterWithBackend` asserting `frame_count == 1` — the M dispatch is inside the op, no recompilation on the token count.
- Standalone build of the modified `.cu` against CUTLASS v4.7.1 on a GB10 (SM 12.1), compared with the unmodified source built identically: the table above (48/48 bit-identical).

## Test result

GB10 (SM 12.1): 48/48 bit-identical, throughput table above. Not run on an SM 12.0 part; by construction those are not chunked unless the weight exceeds their L2.

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
