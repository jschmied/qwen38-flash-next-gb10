DRAFT — needs the user's go. ashhart/TensorFold #548 comment (2026-10-11).

A small follow-up to #633 that also starts the format modules I proposed for `core/quant`. The float64 decoders for FP8 and NVFP4 currently exist in three places: `tests/cuda/quant_ref_tests.zig` (`e4m3`, `e2m1`), `tests/cuda/modelopt_tests.zig`, and `cli/cuda_precision.zig` (which also has `toE4m3`).

What I'd add:
- `core/quant/fp8.zig`: `e4m3`, `toE4m3`, and `reference(h, out []f64)` for block-FP8 (128 x 128 `weight_scale_inv`).
- `core/quant/nvfp4.zig`: `e2m1` and `reference(h, out []f64)` for ModelOpt NVFP4 (uint8 [N, K/2], an e4m3 scale every 16 inputs, a global scale).

Both mirror the existing `mlx.reference` and `dense.reference`. The three copies would call them, with a host test against hand-decoded codes. The CUDA packers in `cuda/fp8.zig` and `cuda/nvfp4.zig` stay where they are, since their layouts are CUDA's.

This is host-only and backend-neutral, so a Metal or HIP port of those formats could check against the same references. I'll open it as one small PR unless you'd rather `core/quant` stay MLX/dense-only for now.
