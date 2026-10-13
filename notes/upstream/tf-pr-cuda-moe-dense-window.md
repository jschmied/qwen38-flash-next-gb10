# TF PR draft: cuda window_attention + moe_glue + dense_rows (operator-only, stacked on #746) - NOT posted

Branch cuda-moe-dense-window (worktree ~/git/tf-kglue), commit 2710ae2d on #746's 4d596fe4. Not pushed.

Title: cuda: sliding-window attention, sigmoid top-k routing and combine, dense rows for every family, checked against float64

## What this changes

Stacked on #746 (its `core/attention_ref.zig`). Operator-only, no model calls them yet:

- `window_attention.cu/.zig`: sliding-window attention, decode rows over a ring of keys (split by key tile, merged
  in tile order) and prompt rows, grouped KV heads, window 0 = full.
- `moe_glue.cu/.zig`: top-k by logit + bias (ties: lower id), sigmoid weights, optionally normalised and scaled,
  a shared-expert slot; a warp form up to 512 experts, a block form beyond; the weighted combine of the slots.
- `dense_rows.cu/.zig`: bf16 rows times bf16 weight rows into fp32 (routers, output heads), a row's bits at any row
  count. `core/moe_ref.zig`, `core/dense_ref.zig`: the float64 references.

Users: families with sliding-window layers or sigmoid-routed MoE on CUDA (Kolibri 1's shapes in the tests; GLM and
DeepSeek-style routing are the same top-k form).

## Receipt

- `tf-cuda-test window`: 48/4x128 window 513 over rings of 640 and 600 (a ring end inside a tile), 8/2x64 window 100
  with 3 streams, 16/1x128 no window; prompt rows from position 0 and across the ring's wrap: worst 0.465, bits at
  any row count and chunk.
- `tf-cuda-test moe-glue`: 384 experts top 6 + shared, 256 top 8 normalised x2.5, 512 top 32 (warp form's widest),
  513 top 6 (block form's narrowest), 4096 top 64; picks exact, weights and sums worst 0.996.
- `tf-cuda-test dense-rows`: [384, 2560], [4103, 2560], [257, 7168]: worst 0.007, bits at every row count.
- Mutations: window bound one key wider (74x), sigmoid exponent x1.0625 (5,527x), a dense row's first product
  dropped (17,529x).
- sm_120: BobClawblaw ran `window`, `dense-rows`, `moe-glue` on an RTX 5090, all passing with mutations (#608).
- Tests: 27/27 steps, 350/350; fmt clean; lean_check 538 as on #746's base.
