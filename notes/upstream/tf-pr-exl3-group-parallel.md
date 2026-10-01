DRAFT — needs the user's go (given 2026-10-01 "ok" to the plan; open after the end-to-end A/B). ashhart/TensorFold PR,
base `pr-141-0.6.1`, head `jschmied:exl3-group-parallel`, stacked on #184.
Title: CUDA EXL3 experts: group the routed experts in parallel (prompts −<X> % on Flash Next, stacked on #184)

Stacked on #184; the new commit is the last one.

The grouping runs as one 1,024-thread block. Each thread scans all R · slots picks for each of its four experts, with
the picks in shared memory. On a 1,024-row Flash Next window that's 2.5 ms, 13 % of `routed()` after #184, and it
grows with the window (5.1 ms at 2,048).

It is now three launches:
1. Per-expert counts, with integer atomics (exact in any order).
2. One block's scan over the experts for places and tiles: the same two scans as before, now on the counts.
3. A warp per expert that compacts its picks in row order with ballots.

The outputs are the same as before: expert ids in id order, members in row order with −1 after, and the tile list. So
everything downstream is bit-identical. No picks go through shared memory any more, so the grouping has no window limit.

## Measured

One GB10, Flash Next EXL3 3.05 bpw (512 routed + the shared expert, 11 slots), alternating runs:

| | #184 | this PR |
|---|---|---|
| grouping, 1,024 / 2,048 rows (ms) | 2.53 / 5.07 | 0.083 / 0.16 |
| `routed()`, 1,024 rows (ms) | 18.71 / 18.88 | 16.35 / 16.38 |
| 8k prompt, prefill (s) | <A> | <B> |
| 32k prompt, prefill (s) | <C> | <D> |
| `routed()`, 1 / 8 / 16 rows (µs) | <E> | <F> |

`routed()` hashes are identical at 1, 8, 64 and 1,024 rows, and so are the first 16 tokens after both prompts.

## Tests

- `tests/cuda/test_group_kernel_smem.py`: the calls pass the count and place buffers. The launch the shared-memory
  ceiling used to refuse now groups exactly, with the tile list checked as well.
- On GB10: test_exl3_experts, test_qwen4_exp_exl3 and test_group_kernel_smem: <RESULT>.

Written with AI assistance (Claude Code); the author reviewed every change.
