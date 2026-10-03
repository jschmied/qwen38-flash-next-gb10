POSTED 2026-10-03 https://github.com/ashhart/TensorFold/pull/300#issuecomment-5970277660. TensorFold PR #300 comment.
Ran the GPU test you couldn't, on one DGX Spark (GB10, sm_121, driver 580.178.04), PyTorch 2.13.0 + CUDA 13.0, at `5cbe389`:

- `tests/cuda/test_cuda_rowgraphs.py`: 2 passed.
- The new CPU tests (`test_cuda_{kvpool,sessions,drafting,memory_admission,lanes,lane_link,tables,rowgraphs_keys}.py`) on the same box (aarch64): 133 passed.

Written with AI assistance (Claude Code).
