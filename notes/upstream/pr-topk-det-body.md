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

| rows | n | stock | this PR |
| --- | --- | --- | --- |
| 1 | 1,024 (k=512) | 8.3 | 10.4 |
| 1 | 8,192 | 10.3 | 30.1 |
| 1 | 32,768 | 18.5 | 37.5 |
| 64 | 8,192 | 18.5 | 57.5 |
| 64 | 32,768 | 55.5 | 127.1 |

1.3–3× per call. Model-level (12 QSA layers) this is ≈ +1.5 % per decode step at 32k context and ≈ +2 % TTFT at 8k tokens by estimate; the end-to-end A/B is running and will be posted here. For comparison, replacing the kernel with `torch.topk` costs +6 % TTFT on the same box.

## Test plan

- New tests in `tests/kernels/test_top_k_per_row.py`: `test_persistent_topk_deterministic` (rows {1, 8, 64} × lengths {1k, 4k, 8k, 20k, 40k} × k {512, 2048} × {random, tie-heavy}: 6 calls bit-identical and equal to the exact reference), `test_persistent_topk_all_equal` (all keys equal → exactly `[0, k)` 20×), `test_persistent_topk_pivot_ties` (tie populations of 2047 … 16385).
- The same 177 cases were run against a standalone build of these exact sources on a GB10 (sm_121): 177 / 177 pass; on the same inputs the unmodified kernel reproduces its own output in 0 / 177 cases.
- Hardware other than sm_121 not tested by me; the change is architecture-independent (no `atomicAdd` slot assignment remains on any path).

## Test result

Standalone harness on sm_121: 177 / 177. The pytest file against the built kernel is queued on the same box; the result and the end-to-end decode / TTFT numbers follow as a comment.

Fixes #54521. Related: #53142, #54912 (QSA ring bound).

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
