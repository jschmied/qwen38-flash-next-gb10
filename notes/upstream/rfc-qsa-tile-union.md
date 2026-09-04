### Motivation.

`_qsa_sparse_paged_gqa_splitk_kernel` (`vllm/models/qwen4_exp/nvidia/ops/qsa.py`) runs one program per (query row, kv head): each row gathers its own `token_topk` (2,048) K/V tokens and runs the GQA dot at M = 16 (12 heads padded). Two things about prefill make that expensive:

- consecutive query rows in a prefill chunk select almost the same blocks — on real selection dumps from Qwen3.8-Flash-Next at 8k, neighbouring rows share 87–94 % of their compressed blocks (Jaccard), so each block is gathered from paged memory ~R times for R rows;
- M = 16 is far below the tensor-core roofline on every part; on GB10 (sm_121) the dot runs at ~12 TFLOPS in this kernel.

At 8k context the QSA attention is ~13 % of TTFT on a GB10 with the current main kernel (#54873 included), more with FP8 KV where the per-tile dequant is repeated per row as well.

### Proposed Change.

A **tile-union** prefill path for `qsa_sparse_paged_attention`: a tile is R consecutive query rows; one program per (tile, kv head) iterates the *union* of the tile's selected blocks, gathers each block once, and applies a per-row membership mask inside the online softmax so every row still attends exactly its own selection. Numerics are those of the stock kernel up to summation order (outputs within 1e-3 bf16 on peaked-softmax tests; a negative control in the test moves the output by >5e-2 for a single swapped block).

Precompute, per call (~0.8 ms for a 4k-row chunk on GB10):
1. from the indexer's `block_indices` / `logical_positions` / `visible_blocks` (no re-parse of the expanded buffer), pack `(block_id * 8 + row_in_tile)` per tile and `torch.sort` (exact width R × block_topk = 1,024);
2. one Triton kernel flags first occurrences, prefix-sums the union position, scatters the union ids as physical page × PAGE + offset (block-table lookup done here, not in the attention loop), writes the int8 `[R, union]` membership matrix and the ≤ CR−1 causal-tail tokens per row (same rule as `expand_qsa_block_indices`).

Attention kernel: pass 1 over the union in steps of BNB blocks (BN = BNB × CR tokens), expanding the CR tokens per block in-kernel; pass 2 a 16-column tile for the causal tails; rows with an invalid request id are masked and written as zeros like the stock kernel; the tile's block-table row is taken from any valid row of the tile.

**Measured on GB10 (sm_121, 24 MiB L2, 99 KiB smem/block), Qwen3.8-Flash-Next, `0.28.1rc1.dev401` + #54873:**

| chunk | stock kernel | tile-union, whole path (precompute + kernel) |
| --- | --- | --- |
| 8k prefill, 3,813 rows | 14.6 ms | 6.5 ms (**2.24×**) |
| 30k chunk, 3,407 rows at 7.5k ctx | 14.2 ms | 8.3 ms (**1.71×**) |
| 30k tail, 283 rows at 12k ctx | 1.19 ms | 0.84 ms (1.41×) |

Server level (vLLM serve, batch 4096, prefix cache off, two starts per arm, all medians of three):

| TTFT | union (R=2) | stock |
| --- | --- | --- |
| 7,503 tokens | 2.60 / 2.59 s | 2.72 / 2.74 s (**−4.7 %**) |
| 29,263 tokens | 10.17 / 10.18 s | 10.57 / 10.57 s (**−3.7 %**) |

Tile: R = 2 rows (M = 32), BN = 32 tokens, 4 warps, 1 stage. On this part BN = 128 does not fit shared memory and R = 4 loses at long context because its union widens to 1.4× a row; a cost model `kernel ≈ tiles × union_blocks × c_R` (c_4 = 12.8 ns, c_2 = 9.1 ns per tile-block) fits all six standalone cells to 5 % and predicts the crossover. Two forms were bisected and rejected on the way: an R-bit membership *bitmask* per union block (spills at M = 64, 60× slower at BN = 32) and re-splitting the expanded index buffer (1.2 ms of torch per call).

Design questions for the maintainers, in the order they decide the shape of a PR:

1. **Separate path or a change to the stock kernel?** This is a second prefill kernel behind an eligibility gate (prefill-sized chunk, `block_topk` a power of two, `PAGE % CR == 0`); the alternative is to add row tiling to the existing kernel. I would take guidance here before writing more.
2. **Data path from the indexer.** The union wants `block_indices`, `logical_positions` and `visible_blocks` *before* `expand_qsa_block_indices`; today the expansion happens inside `QSAIndexer.forward` and only the expanded buffer reaches the attention op. Proposal: the indexer returns (or stores in the QSA metadata) the compact selection alongside the expanded one.
3. **Request boundaries.** Tiles must not straddle requests, so the current gate accepts single-request batches only. The full version pads tile boundaries at `query_start_loc`; this is the main functional gap before a PR.
4. **Tuning on larger parts.** R, BN and warps are GB10 numbers; a GB300/H100 has 228 KiB smem and will want a different tile. I cannot measure that here.

### Feedback Period.

One week (until 2026-09-11), then a PR against the outcome of question 1.

### CC List.

@gau-nernst @peakcrosser7

### Any Other Things.

Code (a self-contained Python/Triton patch on top of `0.28.1rc1.dev401`; the PR will be a real diff against `main`, see the plan below), pinned to one commit so line links stay valid:

- kernels and integration — [`tools/main/qsa_union_patch.py`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py): [`_qsa_union_build_kernel`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L45) (union + membership + physical page bases + tails), [`_qsa_union_attn_kernel`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L93) (pass 1 union, pass 2 tails), [`_qsa_union_build_raw`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L256) (from the indexer's selection), [`qsa_sparse_paged_attention_union`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L287), [`qsa_union_eligible`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L314); the owner-side call and the indexer hand-off at [L358](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L358) and [L386](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/main/qsa_union_patch.py#L386);
- asserting test — [`tools/qsa_union_test.py`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/tools/qsa_union_test.py) (negative control, tail lengths 0..CR−1, permuted physical pages, decoy requests, masked rows, both build paths, stale-input fallback);
- measurements — [finding 115](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/notes/prefill-investigation.md?plain=1#L648) (server A/B above), [finding 111](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/notes/prefill-investigation.md?plain=1#L606) (the tile bisect), [finding 114](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/notes/prefill-investigation.md?plain=1#L635) (two server runs that were wrong and why); raw logs under [`notes/data/`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/notes/data) (`qsaunion13.txt`, `qsaunion16.txt`);
- the plan for the PR (real diff, explicit indexer→owner data path, per-request tiles, warmup, contract, SM121-only dispatch with a benchmarking override) — [`notes/upstream/pr-qsa-union-plan.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/169d32e5f46d14d89073be2f154c6f50c4c8cd49/notes/upstream/pr-qsa-union-plan.md).

The code and this text were written with AI assistance (Claude); every number above was measured by me on the hardware named, and I have reviewed the code line by line.

### Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page, which can answer lots of frequently asked questions.
