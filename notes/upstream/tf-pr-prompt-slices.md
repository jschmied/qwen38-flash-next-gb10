POSTED 2026-10-03 as https://github.com/ashhart/TensorFold/pull/303 (user: "put out pr"). TensorFold PR: flashnext-prompt-slices (9d6e314) against main.
## What this changes

Prompt rows of Flash Next's bf16 and NVFP4 Triton matmuls (`_b16mm`, `_fp4mm`) add their K slices in one program, in `_reduce`'s order, instead of writing fp32 partial sums for a `_reduce` launch. The grid also puts column tiles first, so a row tile's programs share its rows of x in L2. Every bit is unchanged.

At a 2,048-row prompt piece, the split form wrote and re-read SK × M × N fp32 per call: 85 MB for the hyper-connection down matrix (N 324, K 10,240, SK 32) and 168 MB for the NVFP4 shared expert's down (2560 × 640, SK 8). That traffic was most of the call's time. With rows going first in the grid, each column tile also re-read the whole input from DRAM once a piece's x outgrew L2. `TF_FUSED_SLICE_ROWS` (default 256) sets the row count from which slices are added in one program. Any value gives the same bits.

## Receipt

- **Environment:** v0.6.3 (`9356df5`) and this branch; PyTorch 2.13 + CUDA 13, Triton 3.7.1. One DGX Spark (GB10, sm_121). Flash Next with NVFP4 routed and shared experts and 128×128 block-FP8 dense projections, from a local ModelOpt conversion. `serve --parallel 4 --context 131072`, one server at a time, alternating v0.6.3, this, #(B), #(B), this, v0.6.3.
- **Exactness:** `bench_concurrent.py --levels 4 --alone --serial`: 0 unequal and 0 failed in every cell (alone 48, serial 10). Kernels: fused output equals the split launch bit for bit, a row alone equals the same row among 300 / 2,048 / 8,192, and the split output equals v0.6.3's (bf16 and fp32 outputs, the shapes above plus 640×2560, 96×2560, 2560×2560 and 2560×6144).
- **Prompt speed** (`prefill_cold.py`, tok/s, two servers each):

  | | 2k | 8k | 16k | 32k | 64k |
  |---|---|---|---|---|---|
  | v0.6.3 | 1,400 / 1,382 | 1,434 / 1,412 | 1,456 / 1,423 | 1,419 / 1,326 | 1,385 / 1,281 |
  | this | 1,561 / 1,569 | 1,615 / 1,613 | 1,620 / 1,619 | 1,613 / 1,519 | 1,535 / 1,536 |

- **Decode speed** (`bench_openai.py`, greedy median tok/s, code / chat): v0.6.3 54.6 / 44.6 and 54.4 / 44.7; this 54.6 / 44.6 and 54.4 / 44.7. Sampled cells: v0.6.3 49.4–50.0 / 43.5, this 50.2–50.4 / 43.2–43.6.
- **Kernel times**, v0.6.3 → this, ms per call (outputs bit-equal):

  | matmul | 2,048 rows | 8,192 rows |
  |---|---|---|
  | bf16 324×10240 (fp32 out, SK 32) | 1.03 → 0.27 | 4.15 → 1.02 |
  | bf16 640×2560 (SK 8) | 0.44 → 0.11 | 1.93 → 0.36 |
  | bf16 2560×6144 (SK 4) | 1.44 → 1.21 | 13.85 → 4.71 |
  | NVFP4 2560×640 (SK 8) | 1.76 → 0.35 | 6.83 → 1.41 |
- **Tests:**
  - `tests/cuda/test_flashnext_nvfp4_kernels.py`, `test_flashnext_nvfp4.py` and `test_flashnext_nvfp4_loader.py`: 74 passed.
  - The new test is `test_prompt_rows_add_their_k_slices_in_one_program_with_the_split_bits`, with 5 cases. It fails if the fused sum order differs from `_reduce`'s.
  - One existing test launches `_fp4mm` directly and now passes the grid in column-first order.
  - Host subset (`-k "flashnext or qwen4_exp or nvfp4 or bf16"`): the same as v0.6.3, 128 passed, the same 4 failures and 2 collection errors.
- **Not run:**
  - Metal: these are CUDA-only files.
  - Other CUDA GPUs.
  - Other Flash Next CUDA checkpoints, for example local-inference-lab's NVFP4 with MXFP8 dense and shared expert. It takes the `_b16mm` hyper-connection change but not the `_fp4mm` shared-expert one. That checkpoint is not on this box.
  - Two Sparks.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo.
- [x] No precision traded for speed. If the bits change, the description says which sums change and why. (No bits change.)
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [ ] No personal data, machine names, internal hosts or local paths. No AI attribution lines. (No personal data or paths. The commit carries our AI co-author trailer and this line says the work was AI-assisted; drop both when landing, as CONTRIBUTING says.)

Written with AI assistance (Claude Code); the author reviewed every change.
