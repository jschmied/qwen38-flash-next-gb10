# #781 research: which page pool to generalise (local, not posted)

2026-10-13, read-only survey of #625 (main), #637 + #686 (DeepSeek-V4.1 pool, open), the HIP Qwen3.5 engine (#655 stack,
open), GLM Metal, CUDA host admission. Post only if the #781 thread picks up.

## Correction to #781's text
`prompt_radix`'s `Pages` (`prompt_cache.zig:61-75`) is NOT a pool: a 4-function vtable (retain, release, holders,
available). Nothing on main implements it; the only implementation is the HIP engine (not on main).

## The pools
- **#637 `sessions.Pool`** (CUDA): 256-token pages (`pool.zig:29`) shared by several caches, each declaring `row_bytes`,
  `ratio` (a row per `ratio` tokens), `split` (TP shards) (`pool.zig:9-23`). DeepSeek: comp.L 584 B fp8 + index_k.L
  132 B per KV source layer, ~1,790 B/token. Host-only book (free sets per residue, refcounts slot/entry holders, null
  page `:236-262`); device side separate (`kv/device.zig:28-55`: tensor per cache + `[slots, pts]` i32 table, dirty
  ranges uploaded `takeDirty :100`). Admit reserves prompt + max_tokens + 64 (`needPages :381`); `ensure` maps lazily
  (`:125`), dry pool = `PoolExhausted`, no pause. Own prompt store (`store.zig`, digest trie `prefix.zig`, `adopt :158`,
  `planSpill :519`, `enforce :480`, NVMe tier). Fixed table buffer -> graphs stable. Family-neutral except TP residues.
- **HIP Qwen3.5** (`qwen3_5/backend/hip/forward/pages.zig`): 64-token pages hard-coded as recurrence chunk (`:9`,
  `kPageTokens`), K+V of every full-attention layer `[layer][head][page][64][hd]`. `Ids` free list with refcounts
  (`:17-71`). Per-stream descriptor buffer (`state.zig:21-26`), plan buffer passes descriptor addresses each round
  (`plan.zig:50,100`) -> graphs keyed by shape only. Promises prompt + max_new + window at admit
  (`hip_lanes.zig:308-311`), `grow` lazily (`state.zig:134`). Pool = default_streams x capacity (`engine.zig:135`).
  Prompt cache = `prompt_radix` via `Pages` (`hip_prefix.zig:33,152`).
- **GLM Metal** (`glm/pool.zig`): not paged; contiguous MLA region per stream in 256-token blocks, first-fit at admit,
  evicts held prefixes (`:89-140`).
- **Fixed sizing today:** CUDA host `cuda_memory.admit :91` room / stream_bytes (`native/cuda.zig:430`), Nemotron
  `Seq.bytes(c, max_len)` (`cuda_state.zig:124`), prompt cache gets the rest (`:440`). Metal qwen27 `slotCost` = full
  KV + ring + 256 MiB (`qwen27_served.zig:139`). #746's attention takes an i64 `[streams, 2]` table of contiguous K/V
  bases - not paged.

## Answers to #781's three questions
1. Core owns the book (ids, refcounts, reservations, admission, prompt-cache coupling); each engine binds the device
   (tensors, table upload, copyPage, read/write). #637 already splits them (`PageStore`, `pagestore.zig`).
2. Base = #637's `Pool` (multi-cache layouts, ratios, quotas, dirty sync, null page, TP residues). `Pages` stays as the
   narrow interface `prompt_radix` consumes; Pool implements it almost directly. Real conflict: two prefix stores
   (`prompt_radix` on main vs `sessions.Store`/`prefix.zig`) - pick one.
3. One page size per pool, declared by the family; power of two, multiple of every cache's kernel chunk, default 256.
   Per-cache granularity already covered by `ratio`.

**Key finding:** neither paged pool grows a stream past its admit reservation (prompt + max_tokens). The win over full
windows comes from reserving prompt + max_tokens instead of --context, not from mid-reply growth. #781 point 4 (pause
youngest) would be new everywhere -> make it optional / later.

## Minimal family declaration
```
StreamLayout { fixed_bytes: u64, page_tokens: u32, caches: []Cache{ name, row_bytes, ratio = 1, split = false } }
```
Engine device binding: allocate (npages + null page), uploadTable(lo, hi), copyPage, read/write. Kernels get tensor base,
table pointer, table row stride, page shift.

## Changes per pool
- #637: move to core; account fixed_bytes (now in the family's budget.zig, `Boot.fixed`); implement `Pages`; generalise
  `sess.admit`; reconcile its trie with `prompt_radix`.
- HIP: core book replaces `Ids`; `Slot.reserve` replaces `ids.reserved`; page size declared (kernel constant too); core
  sizes the pool.
- GLM Metal: MLA kernels need a page table, else stays a degenerate contiguous user.
- CUDA host: `admit` takes a StreamLayout; Nemotron's contiguous `Seq` and #746's per-stream bases need paged addressing.

## Related open items
#686/#635 (#637's consumer, graph cache), #655 (HIP stack), #505 (HIP prefix retention), #750 (Flash-Next --parallel 4
--context 131072 swaps 22 GB vs 81 GiB estimate - direct motivation), #643 (27B too large for 32 GB), #726 (qwen27 CUDA
fp8 KV at 256K), #752 (/slots endpoint).
