## Purpose

`persistent_topk` (the QSA / sparse-indexer block selection, `csrc/libtorch_stable/persistent_topk.cuh`) returns a different result for identical inputs from call to call: the **order** always varies, and when more keys share the threshold value than the candidate buffers hold, the selected **set** varies too. Downstream, the sparse attention sums the selected keys in output order, so greedy decoding of Qwen3.8-Flash-Next forks between identical requests (#54521; bit-level bisection in our thread on #53142: the indexer is the first module whose output differs with identical inputs, and an exact selection makes a 7.5k-token forward bit-identical).

Cause: output slots are handed out by `atomicAdd` in thread-arrival order, and exact-key ties at the last radix round are taken first-come.

## Changes

- **Single-CTA rows** (decode / medium paths, and the float instantiation of the filtered kernel): `det_select_row` — a radix select that rescans the row per key byte, so there are no candidate buffers to truncate and the pivot is exact, followed by one index-ordered block scan that emits everything above the pivot and then the lowest-index elements equal to it. Both groups come out in **ascending index order**, so the row is two sorted runs and a single merge finishes it.
- **Multi-CTA rows** (> `RADIX_THRESHOLD`): the radix rounds are kept; the emission becomes deterministic — per-CTA `>`/`==` counts published before the barrier, slots from a prefix over CTAs, both groups ranked by index. CTA *c* covers a lower index range than CTA *c+1*, so each group is one ascending run across the whole row and CTA 0 merges them.
- A radix pass is skipped when the threshold bin holds exactly the number of slots still unfilled: the remaining key bytes cannot change the answer. The output is then a single run, which the merge returns from immediately.
- The 256-bin suffix sum and the threshold search run in **one warp** (lane-local serial sums, then a Hillis-Steele suffix scan over the 32 lane totals), removing 32 `__syncthreads()` per call. The emission packs its two flags into a single `BlockScan`.
- `RADIX_THRESHOLD` 32768 → 16384: the single-CTA select caches the row's ordered keys at 4 bytes per element, so 32,768 elements want 128 KB against this device's 101,376 B opt-in; rows above the threshold take the multi-CTA path instead. This is a capacity constraint, not a claim that either path is faster — the measured crossover is row-count dependent, see Limitations. *(Corrected 2026-09-08: this bullet previously read "the deterministic multi-CTA path is cheaper than the single-CTA select above 16k", which the Limitations section already contradicted.)*
- Launcher: caps the dynamic shared-memory request at `sharedMemPerBlockOptin − static __shared__` (needed on sm_121, 99 KB opt-in), and applies its `chunk_size >= TopK` check only on the cooperative path.
- Output contract is **ascending index order**, identical across calls, equal to top-k by value desc / index asc.

`topk_histogram_4096.cuh` is unchanged (its float instantiation is no longer reached).

## Cost (GB10 / sm_121, 5 × 50 launches, median µs, ratio to the unmodified kernel)

| rows | n | k | earlier revision | **now** |
| --- | --- | --- | --- | --- |
| 1 | 4,096 | 2048 | 4.31× | **1.20×** |
| 1 | 8,192 | 2048 | 2.95× | **1.00×** |
| 1 | 32,768 | 2048 | 1.82× | **1.10×** |
| 1 | 65,536 | 2048 | 2.36× | **1.40×** |
| 64 | 1,024 | 512 | 1.66× | **0.74×** |
| 64 | 8,192 | 2048 | 3.10× | **1.10×** |
| 64 | 32,768 | 2048 | 2.25× | **1.32×** |
| 24 | 32,768 | 2048 | 3.72× | **2.14×** |

**Whole grid 0.74–2.14×**, against 1.3–4.3× when this PR was opened. The change came from removing
work: the final sort was ordering data the emission had already ordered, the bin scan cost 32 block
syncs per call, the last radix passes frequently cannot change the answer, and the emission can write
straight into final positions. End to end there is no cost — server-level A/B, three starts per arm,
[numbers here](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5536961535).

**One regression, disclosed rather than hidden.** Cells at n ≤ 16,384 with k = 2048 are about 5 %
slower than an intermediate revision of this branch — 8 rows / 16,384 / 2048 is 18.5 µs → 19.4 µs,
reproducible to ±0.1 over three fresh processes. I could not explain it: a build without the
signed-zero fix measures the same, `cuobjdump` shows identical registers (64) and static shared memory
(5,280 B), the launch geometry for that shape is unchanged, and moving the multi-CTA block out of line
made it *worse*. 18 of 43 cells improve in absolute time and the grid range narrows at both ends, so
the trade is net positive, but the 5 % is real.

## Hardware risks a reviewer should weigh

These are properties of the code, not of the measurements, and only the first is new here.

- **The ≥128 KiB filtered path costs 1.1–2.7×, measured on H100 and A100** ([numbers](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5571662949)).
  With `num_rows > 32` and ≥128 KiB the op takes `FilteredTopKRaggedTransform`, whose per-row work is
  now `det_select_row`. GB10 offers 99 KiB and never executes that branch, so I rented the hardware:
  correctness holds on both parts, and the cost is worst at very long rows (n=65,536), where the row
  cannot be cached in shared memory on any current part. Sizing the request from the device rather
  than a 128 KB constant took n=40,000 from 2.41× to 1.74× on H100 and is included here. The affected
  regime is `rows > 32` — large batch and prefill, not the c=1 decode shape.
- **The inter-CTA barrier is a spin-wait under a non-cooperative launch.** Residency is *estimated*
  host-side from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` and capped with headroom. Concurrent
  kernels, MPS, or another stream can invalidate that estimate, and at occupancy 1 the reservation is
  one CTA globally rather than one per SM. This is inherited, but lowering `RADIX_THRESHOLD` to 16,384
  makes the cooperative path reachable more often, so this PR increases the exposure.
- **`RADIX_THRESHOLD = 16384` is GB10-tuned and is not optimal even here.** Measuring the two
  implementations at identical shapes, the crossover is row-count dependent: n ≈ 24,576 at 1–8 rows,
  **never** at 32 rows (the single-CTA select still wins by 8–35 % at 65,536), and ≈ 32,768 at 64 rows.
  No scalar value is right for all of them, and the correct value on another GPU is unknown. Raising it
  back to 32,768 is not a fix — that costs 60–100 % at n = 24,576–32,768 on 1–8 rows.
- **`__launch_bounds__(kThreadsPerBlock, 2)` is silently ignored on this part.** nvcc reports "Value of
  threads per SM ... is out of range" for every instantiation: 1024 × 2 exceeds the 1,536 threads per SM
  this device reports. Pre-existing, and left alone because changing launch bounds needs its own
  measurement, but the occupancy the source asks for is not the one it gets.

## Test plan

- New tests in `tests/kernels/test_top_k_per_row.py`: `test_persistent_topk_deterministic` (rows {1, 8, 64} × lengths {1k, 4k, 8k, 20k, 40k} × k {512, 2048} × {random, tie-heavy}: 6 calls bit-identical and equal to the exact reference), `test_persistent_topk_all_equal` (all keys equal → exactly `[0, k)` 20×), `test_persistent_topk_pivot_ties` (tie populations of 2047 … 16385), `test_persistent_topk_narrow_value_range` (every key in one coarse histogram bin, the #51782 shape: no candidate may be dropped).
- The same shapes were run against a standalone build of these exact sources on a GB10 (sm_121): 210 / 210 pass; on the same inputs the unmodified kernel reproduces its own output in 0 / 210 cases.
- Correctness is verified on **three architectures**: sm_121 (GB10), sm_80 (A100 80GB) and sm_90 (H100 80GB). The last two were rented, with both arms built from source on the box and three starts each, because the `num_rows > 32` filtered path needs ≥128 KiB of opt-in shared memory and GB10 offers 99 KiB — so that path cannot be exercised here at all. Blackwell and ROCm remain untested. No `atomicAdd` slot assignment remains on any path.

## Test result

GB10 / sm_121, TP1, built from this branch's head:

- `pytest tests/kernels/test_top_k_per_row.py -k persistent_topk`: **221 passed, 26 skipped**.
- Standalone harness over the determinism / exactness grid: **210 / 210**. The unmodified kernel reproduces its own output on **0 / 210** of the same inputs.
- End to end, server-level A/B with three starts per arm: no measurable change in decode or TTFT ([numbers here](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5536961535)).
- Not from this PR, but seen while running the file on this box: all 51 `cooperative_topk` cases fail with `cooperative_topk launch failed: invalid argument` (`cooperative_topk.cu:48`). The backend is gated on SM90+, which sm_121 satisfies, but the cluster launch is rejected here. This PR does not touch that op.

Fixes #54521. Fixes #51782 (the candidate buffers whose overflow dropped keys no longer exist; the narrow-range test covers that shape). #53287 explores the same two defects with a different mechanism (wider coarse histogram, exact fallback on overflow, buffered paths kept); this PR removes the buffers instead and also fixes the output order. Related: this is one of three independent defects that together make Qwen3.8-Flash-Next reproducible on GB10 — the others are the FlashInfer CUTLASS MoE fused finalize (#54945, PR #54948) and the align-mode block-size PRs #54076 / #53798; bisection thread in #53142; #54912 (QSA ring bound).

---

This PR includes AI-assisted code (Claude Code). Every line was reviewed by the submitter.







