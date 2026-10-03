POSTED 2026-10-03 as https://github.com/ashhart/TensorFold/pull/304 (user: "put out pr"). TensorFold PR: flashnext-nvfp4-prompt-experts (2fa39a3) against main.
## What this changes

Flash Next's NVFP4 routed experts get a prompt kernel that decodes each FP4 weight block once per 64 pairs, with the same bits.

The current kernel serves prompts and decode alike: a warp owns 32 columns and streams the expert's whole K once per 16 pairs, so the FP4 words, the e4m3 scales and their loads are redone for every 16 pairs. Its two stages live in registers (gate|up: 255 registers with spills, 16.7 % occupancy). Prompt items now hold up to 64 pairs. A CTA of four warps takes four column blocks of an item. Per 32-input group, a four-deep `cp.async` stage holds the item's rows of x and the warps' weight blocks, and each warp decodes its block once and runs its four 16-row tiles from shared memory.

A pair's mma and fma sequence, its K permutation and its epilogue are the decode kernel's, so its bits are the same. The decode kernel is unchanged, and calls under 64 rows stay on it (`TF_NVFP4_STAGED_ROWS`). Independent of #(A).

## Receipt

- **Environment:** v0.6.3 (`9356df5`) and this branch; PyTorch 2.13 + CUDA 13. One DGX Spark (GB10, sm_121). Flash Next with NVFP4 routed and shared experts and 128×128 block-FP8 dense projections, from a local ModelOpt conversion. `serve --parallel 4 --context 131072`, alternating v0.6.3, #(A), this, this, #(A), v0.6.3.
- **Exactness:** `bench_concurrent.py --levels 4 --alone --serial`: 0 unequal and 0 failed in every cell (alone 48, serial 10). Kernels: gate|up SwiGLU and down outputs equal v0.6.3's bit for bit at 2,048 and 8,192 rows (512 experts, top 10, Flash Next shapes), and equal the decode kernel's at 1–512 rows.
- **Prompt speed** (`prefill_cold.py`, tok/s, two servers each):

  | | 2k | 8k | 16k | 32k | 64k |
  |---|---|---|---|---|---|
  | v0.6.3 | 1,400 / 1,382 | 1,434 / 1,412 | 1,456 / 1,423 | 1,419 / 1,326 | 1,385 / 1,281 |
  | this | 1,541 / 1,542 | 1,590 / 1,583 | 1,614 / 1,560 | 1,582 / 1,507 | 1,483 / 1,476 |

- **Decode speed** (`bench_openai.py`, greedy median tok/s, code / chat, a separate session alternating this, v0.6.3,
  v0.6.3, this): v0.6.3 55.1 / 45.1 and 55.0 / 45.1; this 54.8 / 45.0 and 54.8 / 44.8. Sampled: 49.9–50.0 / 43.7–43.9
  against 49.7–50.1 / 43.7. The decode kernel is the same code, and in process a 256-token decode's GPU time is
  3,627.5 ms with 16-pair prompt items and 3,631.1 ms with this kernel's 64-pair ones, the same kernels and counts.
- **Kernel times** (gate|up + down, ms per layer, 512 experts, top 10): 2,048 rows 13.7 → 7.6; 8,192 rows 40.7 → 19.7. With #(A) and `TENSORFOLD_PREFILL_ROWS=8192`, in-process 8K / 32K prompts take 4.11 / 15.9 s, against v0.6.3's 5.70–5.81 / 22.8–23.4 s.
- **Tests:**
  - `tests/cuda/test_flashnext_nvfp4_kernels.py`, `test_flashnext_nvfp4.py` and `test_flashnext_nvfp4_loader.py`: 69 passed.
  - Two existing tests pinned NVFP4 prompt items at 16 pairs. One now pins 64. The other compares the staged kernel's bits with the decode kernel's 16-pair items, and fails if any pair's bits differ.
- **Not run:**
  - Metal: these are CUDA-only files.
  - Other CUDA GPUs.
  - Other Flash Next NVFP4 checkpoints (not on this box).
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
