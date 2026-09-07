DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122 (2026-09-07).

## The ≥128 KiB Filtered path: measured, and it is worse than the rest of the PR

I listed this as an untested risk rather than leaving it for review to find, so here is the answer.
Rented an H100 and ran both arms built from source on the box — this branch, and upstream at the
merge-base `d9105ea8` — so these are ratios, not absolutes. H100 80GB HBM3, sm_90, 132 SMs,
227 KiB opt-in smem, torch 2.13.0+cu130, **three separate runs**, 5 × 50 launches per cell.
Bundle and raw output: [`bench/h100-filtered`](https://github.com/jschmied/qwen38-flash-next-gb10/tree/52a70eeec92707d513a691c5b883f1e59a0f6062/bench/h100-filtered),
[run1](https://github.com/jschmied/qwen38-flash-next-gb10/blob/52a70eeec92707d513a691c5b883f1e59a0f6062/notes/data/filtered-H100-run1.txt) /
[run2](https://github.com/jschmied/qwen38-flash-next-gb10/blob/52a70eeec92707d513a691c5b883f1e59a0f6062/notes/data/filtered-H100-run2.txt) /
[run3](https://github.com/jschmied/qwen38-flash-next-gb10/blob/52a70eeec92707d513a691c5b883f1e59a0f6062/notes/data/filtered-H100-run3.txt).

**Correctness holds.** 48 shapes (rows 48/64 × n {4k, 8k, 20k, 40k} × k {512, 2048} × {random,
tie-heavy, all-equal}), all three runs: this branch is bit-identical over 6 calls and exactly equal
to the reference on every one. Upstream is non-reproducible on every shape, and loses the selected
**set** on every tie-heavy and all-equal case. So the bug this PR fixes is present on Hopper too,
not only on the 99 KiB part I found it on.

**Cost is the problem. 64 rows, ratio to upstream, three starts:**

| n | k=512 | k=2048 |
| --- | --- | --- |
| 4,096 | 1.01 / 1.45 / 1.09 | 1.01 / 1.39 / 1.06 |
| 8,192 | 1.53 / 1.85 / 1.47 | 1.52 / 1.79 / 1.61 |
| 20,000 | 2.24 / 2.14 / 2.13 | 1.64 / 1.66 / 1.64 |
| 40,000 | **2.42 / 2.43 / 2.31** | 2.10 / 2.12 / 2.08 |

The large-n cells are stable to ±0.05; only n=4096 moves between runs.

**Why.** The Filtered path is a large win for the current kernel and a small one for the rescanning
select. Crossing the `rows > 32` switch at n=16,384, upstream goes 17.5 → 11.4 µs while this branch
goes 23.0 → 18.0, so the ratio opens from 1.31× to ~1.56× exactly where the path turns on.

**And the grid in the description does not transfer.** That 0.74–2.14× is GB10, which never executes
this branch. On H100 even *below* the switch (rows ≤ 32, persistent path) I measure 1.31×. I will
correct the description rather than leave a number there that only holds on one part.

**What I think this means.** The affected regime is `rows > 32` — not the c=1 QSA decode shape
(4 query rows at MTP n=3), but large batches and prefill, which is what H100/A100 deployments
actually run. I would rather put this in front of you now than have it found after merge. Options
as I see them, and I have no attachment to which:

1. Keep the deterministic select only where it is cheap and leave `FilteredTopKRaggedTransform` on
   its existing selection for `rows > 32`. That reintroduces the nondeterminism on exactly the
   large-batch path, so it trades the fix away where it may matter most — I do not like it, but it
   is the smallest change.
2. Make the Filtered path deterministic a different way — its per-row work is a different shape from
   the single-CTA case and I have not tried to optimise it; the 2.4× is the first measurement, not a
   floor.
3. Land the correctness fix as is and treat the Filtered cost as a follow-up, if reproducibility on
   large batches is worth 2× on a kernel that is a small part of the step.

I can run more shapes on either H100 or A100 cheaply now that the harness exists — say the word and
name the grid.
