## Purpose

`persistent_topk` (the QSA / sparse-indexer block selection, `csrc/libtorch_stable/persistent_topk.cuh`) returns a different result for identical inputs from call to call: the **order** always varies, and when more keys share the threshold value than the candidate buffers hold, the selected **set** varies too. Downstream, the sparse attention sums the selected keys in output order, so greedy decoding of Qwen3.8-Flash-Next forks between identical requests (#54521; bit-level bisection in our thread on #53142: the indexer is the first module whose output differs with identical inputs, and an exact selection makes a 7.5k-token forward bit-identical).

Cause: output slots are handed out by `atomicAdd` in thread-arrival order, and exact-key ties at the last radix round are taken first-come.

## Changes

- **Single-CTA rows** (decode / medium paths, and the float instantiation of the filtered kernel): `det_select_row` — a radix select that rescans the row per key byte (no candidate buffers, so no truncation and an exact 32-bit pivot), then one index-ordered block scan that emits every key above the pivot and the lowest-index keys equal to it, then sorts the row. Keys are cached in shared memory when they fit.
- **Multi-CTA rows** (> `RADIX_THRESHOLD`): the existing radix rounds are kept; the emission becomes deterministic — per-CTA `>` / `==` counts are published before the barrier, slots come from a prefix over CTAs, `==` keys are ranked by index with a block scan, CTA 0 sorts the finished row. `RadixRowState` grows by two 64-entry arrays (fits the existing 1 MiB workspace).
- `RADIX_THRESHOLD` 32768 → 16384: the deterministic multi-CTA path is cheaper than the single-CTA select above 16k.
- Launcher: caps the dynamic shared-memory request at `sharedMemPerBlockOptin − static __shared__` (needed on sm_121, 99 KB opt-in), asserts `ctas_per_group ≤ 64` and `chunk_size ≥ TopK`.
- Output contract is now **ascending index order**, identical across calls; equal to top-k by value desc, index asc.

`topk_histogram_4096.cuh` is unchanged (its float instantiation is no longer reached).

## Cost (GB10 / sm_121, 5 × 50 launches, median µs, k = 2048)

| rows | n | stock | this PR | ratio |
| --- | --- | --- | --- | --- |
| 1 | 1,024 (k=512) | 8.3 | 10.4 | 1.25× |
| 1 | **4,096** | **6.2** | **26.5** | **4.31×** |
| 1 | 8,192 | 10.3 | 30.1 | 2.92× |
| 1 | 32,768 | 18.5 | 37.5 | 2.03× |
| 64 | **4,096** | **14.5** | **49.3** | **3.41×** |
| 64 | 8,192 | 18.5 | 57.5 | 3.11× |
| 64 | 32,768 | 55.5 | 127.1 | 2.29× |

**1.3–4.3× per call.** (Corrected 2026-09-07: an earlier version of this table omitted the n = 4,096 rows and quoted the range as 1.3–3×. The worst case is 4.3×, not 3×.) The ratio is worst where `k` is a large fraction of the row and falls as the row grows; at the shapes this model actually issues — `n ≈ context / compress_ratio`, so ≈ 8k rows at 32k context with k = 2,048 — it is 2.9–3.1×.

Model-level (12 QSA layers) that estimated to ≈ +1.5 % per decode step at 32k context and ≈ +2 % TTFT at 8k tokens. **The end-to-end measurement has since been done and the estimate was pessimistic: no TTFT change and no per-turn cost, three server starts per arm** — [numbers here](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5536961535). For comparison, replacing the kernel with `torch.topk` costs +6 % TTFT on the same box, and `top_k_per_row_decode` — the faster alternative — [is not deterministic on sm_121](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5565253041) (0 of 56 shapes) and has the same tie-handling defect.

## Test plan

- New tests in `tests/kernels/test_top_k_per_row.py`: `test_persistent_topk_deterministic` (rows {1, 8, 64} × lengths {1k, 4k, 8k, 20k, 40k} × k {512, 2048} × {random, tie-heavy}: 6 calls bit-identical and equal to the exact reference), `test_persistent_topk_all_equal` (all keys equal → exactly `[0, k)` 20×), `test_persistent_topk_pivot_ties` (tie populations of 2047 … 16385), `test_persistent_topk_narrow_value_range` (every key in one coarse histogram bin, the #51782 shape: no candidate may be dropped).
- The same 177 cases were run against a standalone build of these exact sources on a GB10 (sm_121): 177 / 177 pass; on the same inputs the unmodified kernel reproduces its own output in 0 / 177 cases.
- Hardware other than sm_121 not tested by me; the change is architecture-independent (no `atomicAdd` slot assignment remains on any path).

## Test result

Standalone harness on sm_121: 177 / 177. The pytest file against the built kernel is queued on the same box; the result and the end-to-end decode / TTFT numbers follow as a comment.

Fixes #54521. Fixes #51782 (the candidate buffers whose overflow dropped keys no longer exist; the narrow-range test covers that shape). #53287 explores the same two defects with a different mechanism (wider coarse histogram, exact fallback on overflow, buffered paths kept); this PR removes the buffers instead and also fixes the output order. Related: this is one of three independent defects that together make Qwen3.8-Flash-Next reproducible on GB10 — the others are the FlashInfer CUTLASS MoE fused finalize (#54945, PR #54948) and the align-mode block-size PRs #54076 / #53798; bisection thread in #53142; #54912 (QSA ring bound).

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

