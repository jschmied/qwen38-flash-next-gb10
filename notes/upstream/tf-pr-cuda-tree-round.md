# TF PR: cuda/tree_round (#734)

POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/734 (user: "do pr for this")

## What this changes

Metal runs a drafted tree round on the GPU through `core/tree_round.zig`'s contract: one pick per row, each tree's verified path and emitted tokens, then the commit steps that read that result. CUDA has tree attention and the GDN tree kernel, but no verify or commit step, so every CUDA family that drafts a tree has to write its own.

This adds that step as a shared module, `zig/kernels/cuda/tree_round.cu` with `zig/src/cuda/tree_round.zig`:

- `tf_round_bf16_argmax`: each row's first largest finite bf16 logit, at any row count and stride. A row with NaN or ±inf gets no token and the mask `core/tree_round` uses (1, 2, 4).
- `tf_tree_round_match`: `oracle()` for a batch of trees, one thread per tree. Each `Tree{first, rows, budget, eos_first, eos_count}` names its rows, its budget and its EOS ids. Parents are tree-local. Results use `core/tree_round.Result`'s layout, so `decode()` reads them unchanged.
- Commit: `tf_round_keep_rows` (each tree's kept round rows and their count), `tf_round_gather_rows` (those rows of any bf16 buffer, in path order, for hidden rows or drafter taps) and `tf_round_keep_head` (the last kept row's logits, which the next round's first pick reads). Each one ignores a tree whose result is not `ok`.

No family calls it yet. Nemotron's chain path is unchanged. The 27B in #608 and any other tree drafter can take it as it is. `core/root.zig` now exports `tree_round` so the CUDA test runner can check against it.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus this commit; one DGX Spark (GB10, sm_121, driver 580.173.02, CUDA 13). No checkpoint needed.
- Exactness, `tf-cuda-test tree-round`:
  - Picks: 168 rows over 7 vocabularies (1 to 151,936) equal `rowArgmax` bit for bit. That covers ties, NaN, +inf, -inf and all -inf rows. Each row's padding past `vocab` holds a large finite decoy that must never be picked.
  - Match: 4,096 random trees in one launch equal `oracle()` byte for byte, and `decode()` accepts every result. The trees hit every stop (bonus, length, EOS, invalid) and every status, and include full 16-row paths, budget 0, and malformed trees and picks. 64 trees run alone equal their batched results.
  - Commit: kept rows, counts, gathered rows (width 300, not a multiple of the block) and heads equal the host's. Everything outside the kept rows stays untouched.
- Mutations: four changes to the kernels each fail the test: tie order, a skipped EOS id, the keep count at depth 16, and the head row. So does a changed pick on a verified path.
- Speed: none claimed. Nothing calls it yet.
- Tests run: `zig build test` and `zig build test-cuda-host` (25/25 steps, 208/208 tests), all passing. `tf-cuda-test sample` still matches 2,880 of 2,880. `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: Metal (unchanged); no model, since no family is wired to it in this PR.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (No serving path changes.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (Picks are exact; nothing is summed.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
