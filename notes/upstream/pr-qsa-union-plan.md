# Plan: tile-union QSA prefill → mergeable vLLM PR (after RFC #55394)

Base: vLLM `origin/main` 3284af6b (2026-09-04). All changes are Python/Triton — no C++ build; the branch's files can be
copied onto `vllm-venv-fnmain2` for validation on the GB10 (the venv is dev401 + overlay; the four touched files are not
in the overlay). Positioning: *"Tile-union QSA prefill for SM121: −4.7 % 8k / −3.7 % 30k TTFT on DGX Spark; generic
algorithm, SM121-tuned dispatch."* Build fusion and the sub-1,024-row dispatch are follow-ups, not in this PR.

## Steps, in order

### 1. Real diff on main (½ day) — replaces the patch mechanism
Branch `feat/qsa-tile-union-sm121` in `~/git/vllm-fp8chunk` (clone of upstream). Move the v9 code out of
`tools/main/qsa_union_patch.py` into:
- `vllm/models/qwen4_exp/nvidia/ops/qsa.py`: `_qsa_union_build_kernel`, `_qsa_union_attn_kernel`, `_qsa_union_build`,
  `qsa_sparse_paged_attention_union`, `qsa_union_config()` (dispatch), `warmup_qsa_union()`; the gate inside
  `qsa_sparse_paged_attention` after the stock validation (`use_prefill_config` is already there — reuse it: union only
  when `use_prefill_config` is True).
- `vllm/models/qwen4_exp/nvidia/qsa.py` (owner) and `indexer_qsa.py`: the data path of step 2.
- `vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py`: step 4.
- `tests/kernels/attention/test_qsa_tile_union.py`: step 5.
Drop `_qsa_union_split` entirely (the expanded-buffer path is the fallback no more: no compact selection → stock kernel).
Keep the packed-buffer contract of #54873 (`[rows, width + 1]`, trailing count column) untouched.

### 2. Explicit data path instead of the attribute stash (½ day)
- Owner allocates `self.block_indices_buffer: int32 [max_num_tokens, block_topk]` next to `topk_indices_buffer`
  (persistent; the indexer today allocates `block_indices` with `torch.empty` on every call — the buffer removes that).
- `QSAIndexer.forward(hidden, positions, out, block_indices_out=None)` writes the selection into it before
  `expand_qsa_block_indices`. No new return value: the owner already owns both buffers.
- `forward_qsa(..., union_inputs)` where `union_inputs = QSAUnionInputs(block_indices, logical_positions, visible_blocks,
  query_start_loc, num_decode_tokens)` is a small frozen dataclass the owner fills from `self.block_indices_buffer[:n]`
  and the compressed metadata it can read the same way the indexer does (`metadata[self.indexer.compressed_key_cache.prefix]`:
  `logical_positions`, `visible_blocks`, `query_start_loc`, `num_decode_tokens`, `num_decodes`). No pointer checks.
- Reviewer-facing invariant, asserted: `union_inputs.block_indices.shape == (num_tokens, block_topk)`.

### 3. Per-request tiles (½ day + one A/B) — the functional gap
Rows of a request are contiguous; a tile must not straddle two requests. No padding of the batch: a **row→(tile, slot)
map** built from `query_start_loc`:
```
len_q      = qsl[q+1] - qsl[q]                    # prefill requests only
tiles_q    = ceil(len_q / R);  tile_base = exclusive cumsum(tiles_q)      # T = sum tiles_q
tile(row)  = tile_base[req(row)] + (row - qsl[req(row)]) // R
slot(row)  = (row - qsl[req(row)]) % R
tile_row0[t] = first row of tile t                                         # int32 [T]
```
Build: pack `(block*8 + slot)` and scatter rows into `packed[tile]` via `tile(row)` (one `index_copy_`/scatter instead
of the `.view(T, R, E)`); tails and membership use `slot`. Attention kernel: `row = tile_row0[tile] + r_of_m`, and a
per-tile `rows_in_tile` (≤ R) masks the padded slot of an odd-length request. `tile_request` becomes simply
`token_to_req[tile_row0[tile]]` (every row of the tile is that request by construction; the invalid-request masking of
v4.3 stays). Eligibility: `use_prefill_config` and `num_decode_tokens == 0` (decode rows keep the split-K stock kernel;
mixed batches are a follow-up: run union on the prefill slice and stock on the decode slice) and `num_tokens ≥ min_rows`.
Validation on the box: the existing single-request A/B **plus** a two-concurrent-8k-prompt A/B (the new capability).

### 4. Warmup (¼ day)
`warmup_qsa_union(kv_cache, block_table, *, num_query_heads, compress_ratio, block_topk, cfg)` in `ops/qsa.py`:
`.warmup()` of both Triton kernels with `TritonWarmupTensor` for the deployment-constant constexprs (`R, BNB, CR, GP,
GROUP_SIZE, HEAD_DIM, PAGE_SIZE, TAIL_COLS, N`) and `do_not_specialize=["num_rows", "num_requests", "table_width",
"num_cache_blocks"]` on the kernels (as the stock kernel does). Called from `qwen4_exp_qsa_triton_warmup` right after
`warmup_qsa_sparse_paged_attention`, only when `qsa_union_config()` is not None. `torch.sort` needs no warmup.

### 5. Contract and dispatch (¼ day)
- `TAIL_COLS = next_pow2(R * (CR - 1))` (16 for R=2, CR=4), `ROW_BITS = ceil(log2 R)` replaces the hard-coded `*8`, with
  `R ≤ 8` asserted; `N = next_pow2(R * block_topk)` padded with the sentinel instead of refusing non-power-of-two budgets;
  `PAGE % CR == 0` and `block_topk * CR == token_topk` checked once at layer init (fallback to stock with a one-time log).
- Dispatch `qsa_union_config()`: env `VLLM_QSA_TILE_UNION` ∈ {auto (default), 0, 1}. `auto` → enabled only on compute
  capability (12, 1) with the table `{(12, 1): TileUnionConfig(R=2, BNB=8, num_warps=4, min_rows=1024)}`; `1` forces the
  same config on any architecture (benchmarking override, documented as untuned); `0` off. One log line at init with the
  chosen config, one at first use with the path (`raw` / `stock fallback: <reason>`).
- The compress ratio stays derived (CR constexpr), but the test matrix and the PR body state CR=4 as the validated case.

### 6. Test (¼ day)
Port `tools/qsa_union_test.py` to `tests/kernels/attention/test_qsa_tile_union.py`, no dump files: synthetic selections
with a tunable neighbour overlap (Jaccard 0.9) run through the real `expand_qsa_block_indices`, compared with the stock
kernel under a peaked softmax; the negative control stays (one swapped block must move the output > 5e-2); cases: tail
lengths 0..CR−1, permuted physical pages with decoy requests, **multi-request batches with odd lengths** (new), masked
rows, non-power-of-two `block_topk` padding, `PAGE % CR != 0` fallback. Marked for CUDA ≥ SM120 shapes only where needed.

### 7. Validation on the GB10 (2 h box time)
Copy the branch's four Python files onto `vllm-venv-fnmain2` (record the diff to the overlay), run the test, then:
single-request A/B (two starts, `auto` vs `0`), two-concurrent-prompt A/B (two starts), one warm-turn agent loop to
confirm no regression with prefix caching + MTP. Expected: ≥ the v9 numbers at c=1; a first number at c=2.

### 8. PR
Title `[Kernel][Qwen3.8-Flash-Next] Tile-union QSA sparse attention for prefill on SM121`; body = RFC summary + the
step-7 tables + the four answered questions; `git commit -s`, AI-assistance disclosure, `Co-authored-by`. Open after the
RFC's feedback lands or on 09-11, whichever first; if the maintainers prefer tiling inside the stock kernel, steps 1–6
still hold (only the kernel body moves).

## Effort
≈ 2.5 days of work + ~2 h box time, in the order above; steps 2 and 3 are the ones that change behaviour and get their
own A/B before the PR. Not in scope: build fusion (~0.8 ms/call), the < 1,024-row dispatch table, GB300/H100 tuning
(needs hardware we do not have; the override flag lets others measure).

## Bring-up on other architectures (review input 2026-09-04, adopted)

The override `VLLM_QSA_TILE_UNION=R,BNB,warps,min_rows` (branch commit 35ac471f) lets anyone run the matrix below
without code edits. R is the algorithmic choice (union waste vs shared gather) and stays 2 everywhere until a
measurement says otherwise; BN = BNB × CR and warps are the hardware choices. Register file is 64K/SM on CC 9/10/12,
so the R=4 accumulator pressure does not vanish on larger parts; shared memory is 99 KiB/block on CC 12.x vs 227 KiB
on CC 9.0/10.x, which is what makes BN=128 testable there.

| hardware | CC | start | first challengers |
| --- | --- | --- | --- |
| GB10 / DGX Spark | 12.1 | R2 BN32 w4 | validated (finding 111/115) |
| RTX 5090 / RTX PRO Blackwell | 12.0 | R2 BN32 w4 | R2 BN64 w4, R2 BN32 w8, R4 BN64 w4 |
| GB200 / GB300 | 10.0 / 10.3 | R2 BN32 w4 | R2 BN64 w4, R2 BN128 w4, R4 BN64 w4, R2 BN32 w2 |
| H100 / H200 | 9.0 | R2 BN64 w4 | R2 BN32 w4, R2 BN128 w4, R4 BN64 w4/w8 |
| A100 | 8.0 | R2 BN64 w4 | BN32 w4, BN64 w8; BN128 only if the compiled smem fits |
| RTX 4090 / L40S | 8.9 | R2 BN32 w4 | BN64 w4, BN32 w8 |

Six-arm matrix per architecture, on the three standard chunks (8k early, ~7.5k-context chunk, long-context tail)
with the cold-cache setup: A R2 BN32 w4 · B R2 BN64 w4 · C R2 BN32 w8 · D R4 BN64 w4 · E R4 BN64 w8 · F R2 BN128 w4
(where smem permits). `min_rows` start: 512 on SM12.x/Ada, 1024 on H100/A100/GB200/GB300; sweep 256/512/1024/2048
(the GB10's 283-row tail already runs 1.42× at R2, so 1024 is conservative there, but on faster parts the ~0.1 ms
build/launch weighs more). Expected outcome: R2 everywhere, BN/warps per architecture — recorded in
`_TILE_UNION_TABLE` as measurements arrive; the PR ships SM121 only.

## Dataflow review (2026-09-04 late) — applied on the branch, head 8c09f0c5

Done: int64 `logical_positions` contract (the production metadata buffer's dtype — the branch's first server run
silently fell back on this; the test now spies the call on production dtypes), fused Triton pack kernel (one read of
the block ids, no eager temporaries), union built **in place** over the sorted keys (no `uni` buffer), page lookups
for first occurrences only, causal tails resolved to physical tokens in the build (attention kernel has no block
table), `visible_blocks` dropped (top-k pads with −1), int32 layout math with `searchsorted(out_int32)`, `skip_topk`
guard in the owner, power-of-two compress ratio / positive top-k / sentinel-domain / tensor-contract checks in
eligibility and a static guard shared with warmup, padding rows past `query_start_loc[-1]` covered structurally by
the last request's tiles (written as zeros). 12 tests green on the GB10.

Deferred, with reason:
- **Skip `expand_qsa_block_indices` on the union path** — not safe as is: the MTP drafter's QSA layers reuse their
  step-0 *expanded* buffer on later steps (`set_skip_topk(True)` in the proposer, `compact_topk_indices`), so a
  drafter layer that skipped expansion at step 0 would feed garbage to its later split-K steps. Needs the compact
  selection to follow the same compaction/reuse lifecycle first. ~30 MiB of dead writes per eligible layer stay.
- `output.zero_()` in `forward_qsa` (44.7 MiB per layer at 3,813 rows): removable once both kernels are proven to
  write every row; the union kernel now does, the split-K side needs the same audit.
- Shared per-batch tile layout across QSA layers; one model-wide raw workspace instead of a per-layer buffer;
  keys-only sort (torch.sort writes a hidden int64 index tensor, ~15 MiB per call) — profile the build first.
- Fragmented-batch benchmark (1×4096 / 4×1024 / 16×256 / 64×64) — evidence for the `min_rows_per_request=64` gate.

## PR #55430 — state 2026-09-05 08:40

Open, two commits (`d65244b8` kernel/integration, `798b24e4` tests) on main 8369affa, nine files. CodeRabbit's one
finding (non-power-of-two pad width in the pack kernel's `tl.arange`) fixed and covered by a 1,280-token-budget test;
vLLM's pre-commit (incl. mypy) passes locally; CI's pre-commit waits for a maintainer's `ready` label. The metadata
class carries the shared per-forward layout as an explicit optional field. Smoke serve of the final head
(`notes/data/tusmoke-798b24e4.txt`): 2.59 s at 7.5k, 10.14 s at 29k, dispatch/warmup/path lines present — identical
to findings 117/118. Fragmented batches: finding 119 (neutral). Open on the maintainer's side: the prefill/decode
structural split (#54513 style), dropping the env knob, whatever the kernel-design concern is.

