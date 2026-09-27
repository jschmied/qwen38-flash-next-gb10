# vllm#54458 for Flash-Next warm turns: design memo (2026-09-27, main a9eafde59)

**Bottom line.** Upstream main already has what agent loops need: a Mamba *prompt-tail* checkpoint with a
fine-grained hit and copy-on-write (CoW) (RFC #45702, closed 09-08).
- It is **off for us**: the hash unit is the gcd of the prefix-cacheable groups, 1728; the QSA ring (block 12)
  does not count.
- Turning it on needs `--prefix-match-unit 64` plus the 1-line fix #58368. Our prod base 1ea7c63f4 lacks it,
  and without it every turn is split and nothing registers.
- No allocator change is needed. Options A and B fix net-snix's concurrency collapse, not our recompute.

## 1. Thread and landscape
- **#54458** (open, 08-30): GLM-5.3-Flash, 7808-token blocks, 23 pages/request. We commented 08-30; no reply, no PR.
- **Merged.**
  - `prefix_match_unit` (config/cache.py:91, CLI arg_utils.py:1370) and the Mamba partial tail with CoW and
    the scheduler stop: #49502 (07-26), #50507 (08-05), #51843 (08-12), #55178 (09-05), #53945 (09-08),
    #57382 (09-20).
  - **#58368 (09-25, d5051abaf)** fixes an MTP regression from #55390 (09-21): hit rate 89.8→95.6 % on
    multi-turn Qwen3.5.
  - `prefix_cache_retention_interval` default 0 (#52216, 08-17).
- **Open.**
  - #50551 (09-11, merge conflicts): decode-end state.
  - #57329 (09-23): Mamba2 internal prefill checkpoint. #56960 (09-22): the same for KDA; K3's version is
    #52789/#53614.
  - #55873 (09-24): checkpoint token.
  - #54076 (09-23): moot for us, since engine/core.py:343-352 now takes the minimum over prefix-cacheable
    groups. #55533 (09-17): MTP concurrency cap.
  - #53142/#53798/#55601: MRV2 seeding with `cache_config.block_size` (mamba_hybrid.py:117); harmless at 1728.
    #53558 (09-25): KVCacheConfigBuilder hook, relevant for B.
- **Nobody implements A or B.** #24448 (closed 2025-09) only split attention kernel blocks (worker/utils.py:486-496).

## 2. Code map (main)
- **(a) Block-size derivation.**
  - interface.py:835: `attn_block = align*cdiv(mamba_page, align*attn_B/tok)` (:979-986), raised at
    :987-994, `mamba_block_size = block_size` in align mode (:996), page padding at :999-1016.
  - kv_cache_utils.py:1347 pads Mamba pages. :705-800 resolves scheduler block = lcm and hash unit =
    `prefix_match_unit` or gcd. Our log `[1600,1600,1600,1600,8,1600]` gives a hash unit of 1600.
- **(b) Allocation.**
  - One `BlockPool` (kv_cache_coordinator.py:96). The groups alias one buffer with one `bytes_per_block`
    (kv_cache_utils.py:1570, :1739-1790): a block id is the same slot in every group.
  - The align-mode MambaManager keeps 1 + spec + checkpoint + CoW live blocks (single_type_kv_cache_manager.py:1706-1926).
- **(c) Checkpoints and hits.**
  - scheduler.py:409-523 stops chunks at the block floor, plus at `floor(P/u)*u` when `mamba_partial_cache_hit`
    is set (:369-374, :472-480, :503-521). The flag is on when hash unit < Mamba block (kv_cache_coordinator.py:680-712).
  - `_cache_partial_tail_block` (:2031-2118) keys the tail; CoW runs at :1884-1924; hit probe :1482-1559.
- **(d) Kernel indexing.**
  - `as_strided` views with stride = padded page (kv_cache_interface.py:353-415); align state_indices =
    the last 1 + spec columns (attention/backends/utils.py:1156); columns from mamba_attn.py:416-457.
  - Copies use `base + id*stride` (worker/mamba_utils.py:191-365). Under block-outer layout, k ids are not
    k contiguous pages.

## 3. Options (fp32 GDN state: 48x128x128x4 B = 3 MiB/layer, x36 ≈ 108 MiB per state set)
- **A. One state spans k pages.**
  - Needs k-contiguous runs (fragmentation) and a layer-outer layout or gathers.
  - Touches MambaSpec math, both runners, copy kernels and connectors: 1-2 kLOC.
  - KV cost 0. For us it only moves checkpoint placement, which D1 already does.
- **B. Per-group pools or page sizes.** Needs a fixed memory split and breaks "one id, every group"
  (coordinator, hit reconciliation, connectors). Blast radius like A.
- **C. Several states per page.** Infeasible: one page = one state; a sub-block stride = "all" mode at k× memory.
- **D1. Prompt tail (exists).**
  - Resume at floor(P_prev/u)*u; recompute ≈ new + u/2 ≈ **275 instead of 1026 (-73 %)**.
  - KV cost: +1 Mamba block set per retained turn (≈108 MiB, LRU) and 1 transient CoW block per request.
  - Price: one extra forward at T (≤u tokens; estimated 40-80 ms, **not measured**).
  - Finding-141 turns with 246-339 recomputed tokens ran 0.32-0.38 s, so the median should drop from
    0.59 s to about 0.38-0.45 s. The fp32 state is copied bitwise, so precision is unchanged.
- **D2. Decode-end state (#50551).** Only the reply share of "new", only if re-rendered identically. Measure first.
- **D3. GDN internal prefill checkpoint (the analogue of #57329/#53614).**
  - Removes D1's extra forward.
  - FLA's per-chunk `h` is **bf16** (third_party/flash_linear_attention/ops/chunk_delta_h.py:352), so it
    needs an extra fp32 store. Infrastructure: kv_cache_interface.py:1035-1040, 1112-1140; layers/mamba/checkpoint.py:21.

## 4. Recommendation
1. **Now.**
   - Cherry-pick #58368 (1 line, `use_eagle`→`drop_eagle_checkpoint_block`, :2081 in 1ea7c63f4). With
     `FN_SPEC_NODROP=1`, 1ea7 splits at T but registers at T-u, so nothing registers.
   - Add `--prefix-match-unit 64`: it divides 1728 and is a multiple of the QSA compress ratio 4 (startup
     check at kv_cache_utils.py:779-798).
   - Why 64: a hit needs P_prev mod u ≥ g (generation-prompt tokens not reproduced by the next render),
     which holds on ≥92 % of turns at g ≤ 5, against about 70 % at u=16.
   - A/B: `gb10-ab` warm-turn replay, paired per-turn total, `prefix_cache_hits_total` deltas. Gate: bit-exact greedy output against a cold run with the same split, covering RecoverSSM/replayssm
     CoW and the PLE conv group.
2. **Upstream if the split forward costs ≥40 ms: D3**, ~200-300 LOC.
   - Files: qwen_gdn_linear_attn.py (spec + offsets), gdn_attn.py, chunk_delta_h.py (fp32 store), plus the
     conv window through the checkpoint writer.
   - Tests: the kernel state at T equals `final_state` cut at T, bitwise; a manager test in the style of
     tests/v1/core/prefix_cache/test_mamba_eagle_resume_checkpoint.py; an e2e test in the style of
     tests/models/language/generation/test_hybrid.py:770 with GDN and `prefix_match_unit`. No GDN
     partial-tail e2e test exists today.
3. **#54458 reply, after measuring:** tail hits cover the prefix-cache half. Admission still needs A or B,
   which is out of scope for us.
