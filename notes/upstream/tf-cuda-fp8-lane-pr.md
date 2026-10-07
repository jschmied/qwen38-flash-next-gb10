DRAFT — user's go 2026-10-07 ("ok"), post only with the GPU receipt filled in. ashhart/TensorFold PR against main.
Title: cuda: block-FP8 projections on the Python lane matmul's FP8G kernels

## What this changes

The Zig CUDA engine reads MLX 4-bit only. Block FP8 (128 x 128 e4m3, fp32 `weight_scale_inv`) is how most official FP8 checkpoints ship, and Flash Next's NVFP4 exports carry such layers too. This PR adds it model-free:

- `zig/kernels/cuda/fp8_lane.cu`: the device code of the Python lane matmul (`nvfp4/qmmf.cu`, lines 14-298, comments and ATen dropped, the same arithmetic), with the FP8G instances the Python host launches named: row tiles 16, 32 and 64, each with and without a cluster, the fused prompt kernel, and the slice reduce.
- `zig/src/cuda/fp8.zig`: the Python host's choice of kernel from the shapes alone (`split_k`, `bucket`, fused from 256 rows, slices meeting in a cluster on sm_90 on), the repack into the kernel's fragment order and scale tiles (`_fragment_order`, `from_rows`), and `Lane.matmul`.
- `tf-cuda-test fp8-lane <dir>` with `zig/tests/cuda/oracle/fp8_lane.py`: random block-FP8 weights and bf16 rows through the Python kernel, then through ours.

No family uses it yet; Kolibri-1 (#ISSUE) is the first.

## Receipt

- Environment: base `main` at 041d14a, head HEAD. Zig 0.17.0, nvcc 13.0 (`-Dnvcc -Dsm=121`), one DGX Spark (GB10, sm_121).
- Exactness: RESULT
- Host tests: `zig build test` passes apart from `direct_io`, which fails on this machine on `main` too (O_DIRECT on the build filesystem).
- `tools/zig/lean_check.py` reports nothing in the new files.
- Not run: other GPUs; the reduce path (no sm_90-less GPU here; on the GB10 the slices always meet in a cluster).
