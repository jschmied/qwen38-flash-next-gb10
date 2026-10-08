POSTED 2026-10-08 as TF #517 (user: "post FP8 dtypes"). TensorFold PR from jschmied:core-safetensors-fp8 into main (2026-10-08).

Title: core: safetensors reads the FP8 dtypes

`core/safetensors.zig` refuses any tensor stored as `F8_E4M3`, `F8_E5M2` or `F8_E8M0` with `UnsupportedDType`, so an FP8 checkpoint fails before any family code runs. `cluster/checkpoint.zig` already knows these three dtypes and sizes them at one byte an element; this brings the loader's parser in line with it.

- `DType` gains `f8_e4m3`, `f8_e5m2` and `f8_e8m0`, one byte an element, parsed from their safetensors names.
- Every `switch` over a tensor's dtype outside the cluster planner (`nemotron/weights.zig`, `flashnext/pack.zig`) has an `else` branch, so existing families refuse FP8 tensors the way they refuse other dtypes they don't read.
- A test parses one entry of each and checks the byte count is still enforced.

First of a small series toward a block-FP8 CUDA family (#481): Kolibri-1's checkpoint stores its weights as `F8_E4M3` with fp32 `weight_scale_inv`.

Receipts: GB10 (DGX Spark), Linux aarch64, Zig 0.17.0, base f8fe17d. `zig build test`: 76/77, the one failure being `direct_io` on `main` too (#486). Mapping Kolibri-1's released checkpoint with this parser reads all 116,303 tensors.
