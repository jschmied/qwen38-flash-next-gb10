POSTED as #242 (2026-10-02, user: "test nvfp4 prefill to the end and do pr"). Head jschmied:qmmf-prompt-slices 5e89fbd.
Title: CUDA NVFP4/FP8 lane matmul: prompt rows add a tile's K slices in one block (Flash Next NVFP4 prompts −5…−7 %, same bits)

`qmmf` splits K by shape (`qmm.split_k`), so a row's bits never depend on M, and meets the slices in a cluster of
SK blocks. That is right for decode windows, but prompt chunks take the same path: on a GB10 the split shapes of Flash
Next's block-FP8 dense layers ran at 22 TFLOPS (6144 → 2560, 8 slices) and 30 TFLOPS (2560 → 512, 4 slices), against
56–64 for the unsplit ones.

Prompt rows (≥ 256, split > 1) now run every slice of a tile in one block (`bm` 0): each slice from zero over its own
groups, added to the running sum in slice order, then scaled. That is the cluster's (and the reduce's) arithmetic, so
the bits are unchanged, with no cluster, no partials and no reduce. Decode windows and unsplit shapes keep their path.

## Measured

One GB10, Flash Next with our ModelOpt export (NVFP4 routed experts, block-FP8 dense layers), v0.6.2 vs this PR
([data](https://github.com/jschmied/qwen38-flash-next-gb10/tree/28258f958e2bb27107e1c056e28daf76ef8925cc/notes/data/tfqmmf)):

| prefill (s), 3 alternating rounds | v0.6.2 | this PR | |
|---|---|---|---|
| 8k | 5.99 / 6.05 / 6.24 | 5.70 / 5.77 / 5.76 | −4.6…−7.6 % |
| 32k | 24.59 / 24.29 / 24.97 | 23.36 / 23.12 / 23.17 | −4.8…−7.2 % |

The first 16 tokens after both prompts are identical. Block-FP8 prompt GEMMs at 2,048 rows: 6144 → 2560 22.5 → 56
TFLOPS, 2560 → 512 30 → 49; output hashes identical on every shape.

## Tests

- `tests/cuda/test_nvfp4_linear.py`: new `test_prompt_rows_add_their_slices_in_one_block_with_the_clusters_bits`
  (FP4, FP8, block FP8 and MXFP8; 8 and 4 slices; 300 rows): equal to an explicit cluster launch on the same rows, and
  16-row decode windows give the same bits.
- On GB10 (sm_121): tests/cuda 1,209 passed, 98 skipped.

Not run: SM 8.9 (no clusters; there the decode path still meets slices through `part` and the reduce, unchanged).

Written with AI assistance (Claude Code); the author reviewed every change.
