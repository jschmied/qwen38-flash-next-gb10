DRAFT PLAN (nothing posted) - shared CUDA kernel PRs, once the throttle lifts (#654 / #669 reviewed)

Branch stack on the fork: cuda-norms -> cuda-moe-glue -> cuda-dense-rows -> cuda-qk-rows -> cuda-window-attention
-> cuda-tree-attention -> cuda-qwen-glue (act_rows, gdn_prep, embed_rows, block_attention, qk strides, tree d256 +
shared groups). Each module: zig/kernels/cuda/X.cu, zig/src/cuda/X.zig, a float64 reference in zig/src/core, a
`tf-cuda-test` command (tolerance, bits at 1/3/17/... rows, a mutation it catches).

Consumers (the point of each PR):
- On main: Nemotron CUDA (tf_nemo_add_rmsnorm, tf_nemo_embed duplicate add_rms_h / embed_rows).
- #608 Qwen3.8-27B: every Triton glue kernel (target + DFlash2 drafter) -> fully Triton-free (fork q27-glue).
- Kolibri 1 (not upstream; fork kolibri1-forward): six modules.

Order (one topic each, the next only after the previous lands):
1. POSTED as #736 (embed only moved; Nemotron norms kept). norms + embed_rows, with Nemotron moved onto them (same bits if the shared kernel keeps Nemotron's sum order;
   measure speed first). Smallest, has a consumer on main.
2. act_rows + gdn_prep + qk_rows (strided/inv/bf16 norms): the 27B's per-row glue - lands with or after #608.
3. tree_attention (d128/256, shared committed groups) + block_attention: the 27B's attention and drafter.
4. moe_glue, dense_rows, window_attention: with the Kolibri family.

Evidence to carry (27B, our GB10): serial 2/4 replies equal 0.6.6, the other two differ at bf16 near-ties (top-2
17.0 vs 17.0; 23.25 vs 23.125); drafted == serial; bench_concurrent --alone --serial 12/12 cells equal; teacher-forced
NLL 0.8058 vs 0.8029, KL 0.0012, argmax 99.12 %; attention 46.1 vs Triton 49.0 ms on the drafted long prompt.
Not bit-exact with 0.6.6: the maintainer's call (#608 asked for kernels "gated against your captures").
