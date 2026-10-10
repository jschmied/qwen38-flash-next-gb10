POSTED 2026-10-10 as TF #548 comment (user: "yes"). https://github.com/ashhart/TensorFold/issues/548#issuecomment-6101147252

@BlivionIaG, #612 and #613 and our CUDA pieces are converging on the same layer from two sides, and I'd like them to fit together before either hardens:

- `core/quant` (#612): reading a checkpoint's weights, detecting the format, cutting them for tensor parallelism, uploading. Backend-neutral.
- `cuda/qlinear.zig` (#587, #595) and `cuda/experts.zig` (#605, #629): the CUDA side. Each format's device layout (repacked tiles), its kernels, and one decode-row and one prompt-row call per format.

A split that keeps one of each:

1. **`core/quant` owns the host side for every backend**: formats, `Config`, TP cuts, the `Uploader`. We'd add the two formats CUDA serves today as format modules there: block-FP8 (e4m3 with 128 x 128 `weight_scale_inv`) and NVFP4 (ModelOpt: uint8 [N, K/2], e4m3 [N, K/16], `weight_scale_2`). Each comes with its exactness test in the form of `quant/exact_test.zig`.
2. **The device layout and kernels stay per backend.** CUDA's lane tiles and HIP's tiles differ, so `qlinear` / `experts` would take a `core/quant` weight (its `Buf`s and format) and repack it on the device, as `Affine4Kernels.pack` does today for MLX. HIP's kernels slice would do the same with its own tiles.
3. **One rule for which kernel runs.** #613's family rule (a row's bits never depend on how many rows share a launch) is the contract our CUDA kernels already keep. Their K slices come from the weight's shape alone (`qmm.split_k`), so a row's bits are the same at every row count; ashhart checked FP8 from 1 to 300 rows on #587. Registering CUDA's entries in the same registry can come after; nothing in the split depends on it.

If that's the direction, we'd send (1) as small PRs on top of #612 once it lands, and move `qlinear` / `experts` onto `core/quant` weights after that. Nemotron's bits are checked unchanged at each step, as with #587 and #605. Happy to adjust to whatever shape #491 settles on.
