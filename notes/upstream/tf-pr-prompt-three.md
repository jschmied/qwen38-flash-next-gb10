POSTED 2026-10-03 as https://github.com/ashhart/TensorFold/pull/318 (user: "post the three as one"). TensorFold PR: flashnext-prompt-three (339305c) against main, stacked on #304.
## What this changes

Three small changes to Flash Next's NVFP4 prompt path, each with unchanged bits. The branch sits on #304: its first commit is #304's, and the last four are this PR.

1. **Fused hyper-connection write-back and norm for bf16 HC matrices** (`forward.py`, `46cff3c`). The released `hc_fused.write_norm`, which checks itself byte for byte against the separate kernels at startup, was gated on MLX 4-bit HC matrices. An NVFP4 checkpoint's bf16 ones now take it too, and `_readout_b16` skips the norm it already wrote. A new test fails without the change.
2. **Gate|up prompt experts split a column block over two warps** (`experts.cu`, `df4a4c9`, on #304's kernel). Each warp keeps two of a block's four n8 tiles for both matrices. That halves the accumulators and gives eight warps a CTA, with the same per-pair mma and fma order. Down keeps whole blocks, because splitting it was slower.
3. **Fused prompt rows of the lane matmul take 128×128 tiles** (`qmmf.cu`, `dbd2ec9`). Where the padded width holds whole 128-column tiles, the fused (#242) path runs 128×128 blocks of eight warps with three stages, and the block scales load per 64-column tile. Every output keeps its group order and scaling. The extension name moves to `v5`.

## Receipt

- **Environment:** v0.6.3 + #304 (`2fa39a3`) as the base, and this branch; PyTorch 2.13 + CUDA 13. One DGX Spark (GB10, sm_121). Flash Next with NVFP4 routed and shared experts and 128×128 block-FP8 dense projections, from a local ModelOpt conversion. `serve --parallel 4 --context 131072`, alternating #304, this, this, #304. The branch also merges cleanly onto v0.6.4.
- **Exactness:** `bench_concurrent.py --levels 4 --alone --serial`: 0 unequal and 0 failed in every cell, for the base and for this. Kernels: the 128×128 tiles give the 64×64 tiles' output bit for bit, at 300 / 2,048 / 8,192 rows on five dense shapes. The split gate|up kernel's outputs equal v0.6.3's at 2,048 and 8,192 rows.
- **Prompt speed** (`prefill_cold.py`, tok/s, default 2,048-row pieces, two servers each):

  | | 2k | 8k | 16k | 32k | 64k |
  |---|---|---|---|---|---|
  | #304 | 1,572 / 1,560 | 1,620 / 1,590 | 1,630 / 1,610 | 1,561 / 1,537 | 1,520 / 1,470 |
  | this | 1,609 / 1,608 | 1,664 / 1,649 | 1,652 / 1,664 | 1,636 / 1,556 | 1,590 / 1,548 |

- **Decode speed** (`bench_openai.py`, greedy median tok/s, code / chat): #304 54.3 / 44.6 and 54.4 / 44.6; this 54.4 / 44.7 and 54.4 / 44.8.
- **In process with `TENSORFOLD_PREFILL_ROWS=8192`** (#303 + #304 against #303 + #304 + this, two rounds, the same first tokens): 8K 4.075 / 4.068 → 3.892 / 3.939 s; 32K 15.76 / 15.91 → 15.24 / 15.37 s.
- **Kernel times:**
  - Hyper-connection write-back and norm for an 8K prompt: 441 → 340 ms of GPU time.
  - Gate|up at 8,192 rows: 13.3 → 11.9 ms a layer.
  - Block-FP8 prompt matmuls at Flash Next's dense shapes, 8,192 rows, summed: 25.7 → 24.2 ms.
- **Tests:**
  - `tests/cuda/test_flashnext_nvfp4_kernels.py`, `test_flashnext_nvfp4.py`, `test_flashnext_nvfp4_loader.py` and `test_nvfp4_linear.py`: 93 passed.
  - `test_flashnext_hc_fused.py` and `test_flashnext_hc_upmix_prefill.py`: 53 passed. That includes the new `test_bf16_hyper_connections_take_the_fused_prompt_write_back`, which fails on #304's `forward.py`.
- **Not run:**
  - Metal: these are CUDA-only files.
  - Other CUDA GPUs. The HC fusion is GB10-only anyway, by its own self-check gate.
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
- [ ] No personal data, machine names, internal hosts or local paths. No AI attribution lines. (No personal data or paths. The commits carry our AI co-author trailer and this line says the work was AI-assisted; drop both when landing, as CONTRIBUTING says.)

Written with AI assistance (Claude Code); the author reviewed every change.
