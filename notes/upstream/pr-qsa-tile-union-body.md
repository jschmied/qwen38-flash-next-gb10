DRAFT PR body — open on the user's go. gau-nernst asked for the PR on the RFC (2026-09-05 01:38). Branch: https://github.com/jschmied/vllm/compare/main...feat/qsa-tile-union-sm121 (2 commits, 8 files, rebased on 8369affa)

# [Kernel][Qwen3.8-Flash-Next] Tile-union QSA sparse attention for prefill on SM121

Implements RFC #55394.

## Summary

In prefill, consecutive query rows of Qwen3.8-Flash-Next select nearly the same compressed blocks (Jaccard ~0.9 at 8k), yet the split-K QSA kernel gathers every row's selection on its own and runs the GQA dot at M = one head group. This PR adds a **tile-union** prefill path: R consecutive rows of one request form a tile, the kernel iterates the union of the tile's selected blocks, gathers each block once, and applies a per-row membership mask inside the online softmax. Every row still attends exactly its own selection; results match the split-K kernel up to summation order.

**Enabled by default on SM121 only** (the part it is tuned on). `VLLM_QSA_TILE_UNION=1` forces the SM121 tile on any device, `R,BNB,warps,min_rows[,min_rows_per_request]` forces an explicit tile for bring-up elsewhere, `0` disables. No behaviour change for other GPUs, no persistent memory there.

## Measured (GB10 / DGX Spark, sm_121, TP1, this branch vs the same branch with the path disabled, two server starts per arm, medians of three, prefix caching off, `--max-num-batched-tokens 4096`)

| | tile-union | split-K | Δ |
| --- | --- | --- | --- |
| TTFT, 7,503 tokens | 2.58 / 2.57 s | 2.65 / 2.65 s | **−2.8 %** |
| TTFT, 29,263 tokens | 10.13 / 10.09 s | 10.28 / 10.28 s | **−1.7 %** |
| two 8k prompts concurrently, pair wall | 5.20 / 5.19 s | 5.31 / 5.31 s | −2.2 % |
| 30k + 8k concurrently, pair wall | 12.61 / 12.55 s | 12.80 / 12.79 s | −1.6 % |
| 8-turn agent loop, MTP 3 + prefix cache | 1.69 s/turn | 1.72 s/turn | unchanged (below the row gate; decode untouched) |

Kernel-level on real selection dumps: 2.5–2.7× the split-K kernel time on 8k chunks, 1.9× at 7.5k context, 1.4× on a 283-row tail chunk (standalone, whole path incl. the union build).

## Design

- `ops/qsa_tile_union.py`: config/dispatch table, host-side eligibility (metadata only, no device reads), row → tile layout from `query_start_loc` (tiles never straddle requests; computed once per forward and shared by all QSA layers), a fused pack kernel (block ids → sort keys), `torch.sort`, a build kernel that rewrites the sorted keys in place into physical token bases + int8 membership + count and resolves the causal tails, and the attention kernel (union pass + tail pass in one online softmax; no block-table reads).
- Indexer: `block_indices_out` receives the selection before expansion (one workspace per device shared by all layers); `expand=False` skips the expansion for eligible batches, except for layers the MTP proposer marks as reusing their expanded rows (`reuses_selection`).
- Owner: decides eligibility before the indexer runs; inputs for an ineligible batch raise instead of falling back (the expanded buffer may be undefined). Output is zeroed only past `num_tokens` (both kernels write every row, invalid ones as zeros).
- Warmup: the three kernels are compiled through the existing Qwen4Exp QSA warmup hook.
- Tile on SM121: R=2, BN=32 (8 blocks × 4 tokens), 4 warps, gate 1,024 rows and 64 rows per prefill request. Chosen by a 2×2 bisect (membership form × addressing) and an R/BN/warps sweep; R=4 spills at M=64 on this part, BN=128 exceeds the 99 KiB smem.

## Tests

`tests/models/qwen4_exp/test_qsa_tile_union.py` (14 cases, CUDA): single and multi-request batches with odd lengths and a 1-row request, zero-length requests, padding rows past `query_start_loc[-1]` (both synthetic and production-shaped: `token_to_req` 0, position −1, ids −1), a one-page context with padded selections, invalid-request rows at request boundaries, the gate (decode rows, small batch, fragmented batch, env off → error), config parsing and validation, static/tensor contracts, the shared layout, warmup, production dtypes (int64 positions) with a spy asserting the path executed, and a negative control proving the tolerance has power (one swapped block per row moves the output by > 0.05).

## On the RFC feedback (@gau-nernst)

- *Split prefill from decode/spec-decode like #54513* — the path is selected only under `use_prefill_config`; decode and spec-decode rows never see it (batches with decode rows are ineligible and stay on the split-K kernel).
- *Permanent prefill kernel selection if it is always faster* — it is faster on every prefill shape we measured on SM121, so there it is the default prefill kernel; on other parts it is untuned and stays off until someone measures (the override exists for that). Happy to drop the env knob entirely once a second architecture is in the table.
- *Inputs from the metadata* — yes: `logical_positions`, `query_start_loc`, `num_decode_tokens`, `num_prefills` come from the QSA forward metadata; the compact selection comes from the indexer's top-k output before expansion (a per-device workspace), which is not in the metadata today.
- *Multiple requests are a must* — done: tiles are built from `query_start_loc` and never straddle requests; measured with two concurrent prompts above and tested with uneven, 1-row and zero-length requests.

## Not in this PR (follow-ups, in the RFC)

Tiles for batches mixing decode and prefill rows; tuning on SM120 / SM100 / SM90 (the override collects it; table entries need measurements); keys-only segmented sort; skipping the expansion for MTP-reused layers (needs the compact selection to follow the compaction lifecycle).

---

The code and this description were written with AI assistance (Claude); all measurements were run by me on the hardware named, and I reviewed every line.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_011SuBgdp87NbfLbiigmzn1z
