# TF PR draft: cuda/tree_round (HELD — throttle: #730, #731, #733 open)

Branch: jschmied/TensorFold `cuda-tree-round` @ 07674c37 (on main 0efbd447).

## What this changes
- `zig/kernels/cuda/tree_round.cu`: core/tree_round's contract on CUDA —
  `tf_round_bf16_argmax` (256 threads/row, lower id on ties, nonfinite mask 1/2/4),
  `tf_tree_round_match` (one thread per tree, batched: `Tree{first, rows, budget, eos_first, eos_count}`,
  tree-local parents), commit: `tf_round_keep_rows`, `tf_round_gather_rows`, `tf_round_keep_head`.
- `zig/src/cuda/tree_round.zig`: `Kernels.resolve(mod)` + launch wrappers; extern structs with layout test.
- `core/root.zig` exports `tree_round`; the tf-cuda-test runner imports `core`.

## Receipt (GB10, sm_121)
- `tf-cuda-test tree-round`: 168 argmax rows over 7 vocabularies (1..151936, padding decoy past vocab) == rowArgmax;
  4096 batched random trees == oracle() byte-for-byte and decode() accepts (stops bonus/length/eos/invalid
  2255/569/650/536, all 4 statuses, full 16-row chains); 64 trees alone == batched; commit kernels == host.
- 4 kernel mutations caught (tie order, EOS skip, keep count at depth 16, head row).
- `zig build test` green; `sample` still 2880/2880; lean_check 539 (baseline); fmt clean.
- Not run: Metal (no change there); no family wired yet — consumers: #608 27B, Kolibri, DS4.1 DSpark, GLM.

## Test bug found while writing it
Sentinel fills on the legacy stream raced the kernel on a non-blocking stream (0xA5 status at tree 1664) —
fills must go on the kernel's stream.
