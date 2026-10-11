# TensorFold: host code that could be shared across Metal / CUDA / HIP (survey 2026-10-11, main e7c523eb)

Read-only survey, not posted. Only Nemotron has a CUDA port; HIP is runtime + one test kernel. So duplication is
Nemotron Metal vs CUDA, cuda/* vs hip/*, native/metal.zig vs native/cuda.zig, and paired tests.

| # | candidate | files | ~lines | risk |
| --- | --- | --- | --- | --- |
| 1 | learned-change sites, generic over memory | nemotron/adapters.zig vs cuda_sites.zig | 220 | low-med |
| 2 | GPU runtime: Args, Usage, launch geometry, graph | cuda/{launch,memory,graph,abi} vs hip/* | 100-300 | low (args/usage) |
| 3 | native glue: modelContext, slide/prompt-cache wiring, thread-binding backend wrapper | native/metal.zig vs native/cuda.zig | 60-150 | low |
| 4 | trainer host bookkeeping (pass prologue/epilogue, indexing) | nemotron/train.zig vs cuda_train.zig | 90 | low-med |
| 5 | MLX 4-bit host math (residual split, pocket dequant, row/col cuts) | weights.zig/learned.zig vs cuda_weights.zig | 100 | med |
| 6 | GPU sampling rule (Rule, near=20, margin=8) incl. Metal cfg packing + test copies | cuda_sampler/cuda/sample.zig, forward.zig:260, tests | 40 | low |
| 7 | f64 e4m3/e2m1 decoders (3 copies) -> core/quant | tests/cuda/quant_ref_tests, modelopt_tests, cli/cuda_precision | 40 | low |
| 8 | bf16 conversions (~13 copies, NaN handling diverges) | core/*, families/*, tests | 50 | low if bit-exact |
| 9 | train prompt-prefix layout + kernel ABI structs | train_prefix.zig vs cuda_train_prefix.zig | 50 | low |
| 10 | window-cost measure/cache, two cost-table types | timing.zig vs cuda_costs.zig, lanes/cost_cache | 40 | med |
| 11 | HIP affine test: Python fixtures -> Zig f64 references | tests/hip_affine_fixtures.py, hip/affine_gpu_test.zig | 60 + no Python | med |
| 12 | MLX quant-config + index.json parsing re-done per family | core/quant/mlx.zig vs 6 families | 100+ | med |

Checked: the `--context 0` difference (Metal: no window; CUDA: the model's window) is documented as family-specific
in README.md:171, not a bug. Not worth unifying: affine-4 tiling (different layouts), MLX quantize rounding, draft
plans (GPU trees vs MTP chains), MoE routing (device-side), memory admission policies, lane backends, slide learners
(already on shared Learner), checkpoint loaders. core/quant is exported but has no users yet.
Ours-adjacent: #6 extends PR #654; #7 extends #633's references.
