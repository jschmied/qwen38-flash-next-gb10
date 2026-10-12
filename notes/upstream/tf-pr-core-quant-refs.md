POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/730 (user: "yes do 1..3")
TITLE: core/quant: FP8 and NVFP4 values and float64 references in one place

## What this changes

The float64 decoders for FP8 (`e4m3`, `toE4m3`) and NVFP4 (`e2m1`) existed in three copies: `tests/cuda/quant_ref_tests.zig`, `tests/cuda/modelopt_tests.zig` and `cli/cuda_precision.zig`. They now live once, in `core/quant/fp8.zig` and `core/quant/nvfp4.zig`, together with a float64 reference for a block-FP8 stack and an NVFP4 stack. The two tests and `check-precision` decode through them. This is the first piece of the `core/quant` format modules proposed in #548. No kernel and no engine path changes.

`core/exlr8_codec.zig` (1.0.8) has its own f32 `e4m3Value`. That could decode through `fp8.e4m3` too, but I left it for a separate change.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus this commit; one DGX Spark (GB10, sm_121, driver 580.173.02, CUDA 13).
- Exactness: no kernel or engine change. The GPU tests that use the moved decoders give the same results: `tf-cuda-test quant-ref` (every product within 0.510 of its float64 tolerance) and `tf-cuda-test modelopt` (NVFP4 relu^2 experts, rms error up 3.316e-3, down 1.646e-3).
- Decode and prompt speed: unchanged (no kernel, launch or engine path touched).
- Tests run: `zig build test`, 44/44 steps; new host tests "e4m3 codes and their rounding", "a block-FP8 stack against hand-decoded values", "an NVFP4 stack against hand-decoded values". A wrong subnormal exponent in `e4m3` fails 5 of them. `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: Metal (no Metal code touched); RTX cards (no hardware here).

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (No engine change.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (No bits change.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.
