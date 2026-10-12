POSTED 2026-10-12 as TF #608 comment (user: "yes"). https://github.com/ashhart/TensorFold/pull/608#issuecomment-6119312996

On the first point, the glue kernels: I have generic versions of the 27B's set on a branch, written as shared `zig/kernels/cuda` modules rather than qwen27 files, so Nemotron, Kolibri and the 27B can take the same ones. Each has a float64 host reference in `core/`, a `tf-cuda-test` command, bits checked equal at 1, 3, 17 and 64 rows, and a deliberate mutation it catches. All were tested at the 27B's shapes.

| Triton kernel | replacement |
|---|---|
| `_add_rmsnorm` | `norms.addRmsH` (bf16 residual, bf16 weights, 64-value group sums) |
| `_gated_norm` | `norms.gatedRms` |
| `_gate_mul`, `_swiglu`, `_group_sums` | `act_rows` |
| `_embed` | `embed_rows` (MLX affine-4, bf16) |
| `_attn_prep` | `qk_rows.rowsFrom` (strided `[q \| gate]` rows, rotary 64 of 256) |
| `_gdn_pre` | `gdn_prep` (each row's conv window, so trees work) |
| `_paths`, `_tail`, `_merge` | `tree_attention` at head dim 256 |

Branch: https://github.com/jschmied/TensorFold/tree/cuda-qwen-glue (stacked on the shared-layer pieces claimed in #548).

Two gaps. I have not run them inside this engine or against your captures, because I don't have the 27B on my GB10. And `_shared` (a stream's committed keys read once for all of its draft rows) has no counterpart yet; the tree kernel reads them per row, which is exact but slower at long context with wide trees.

@philip-pentatonic, if they're useful, take them into your branch, or tell me and I'll wire them into `qwen27/cuda` there. If you or @ashhart would rather write your own, say so and I'll leave it.
