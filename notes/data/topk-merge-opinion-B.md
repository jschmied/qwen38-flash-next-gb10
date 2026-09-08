## Summary of what I found in the code (not in the description)

Three things change the analysis materially:

1. **#55122 does not "rescan from global."** `det_select_row` (`persistent_topk.cuh:157`) caches the row's ordered keys in shared memory when they fit (`cached` flag, line 175). On GB10 they *always* fit for the single-CTA domain: `sharedMemPerBlockOptin = 101376 B`, `RADIX_THRESHOLD = 16384` → `det_select_row_bytes` = 2176 + 65536 = 67712 B. So passes 1–3 and the emission read **shared**, not global. Its global traffic is *lower* than stock's, and it is still 1.14–1.30x slower. The traffic count therefore does not predict its cost.
2. **`histogram_2048_topk` and `histogram_256_topk` are already dead on the #55122 branch** — defined at lines 352 and 617, called from nowhere (`grep` confirms). The dispatcher at line 1189 sends every row ≤ `RADIX_THRESHOLD` to `det_select_row` and everything above to `radix_topk`. `FilteredTopKUnifiedKernel` (line 1305) also calls `det_select_row`.
3. **Someone already built and measured the obvious merge.** Branch `wip/topk-union-preview`, commit `9384be3950`, "#55122's ordering as a post-pass on #55314's selection": **1.76–1.99x**, rejected as det-166. It bolts `union_reorder_inplace` onto the end of #55314 — a k-gather to recover the pivot, a full global gt-count pass, and a full global emission pass. That is the naive design and it is the worst of the three.

Also: GB10 is **48 SMs, 24 MB L2**. A 16384-float row is 64 KB; 32 rows is 2 MB. The whole working set is L2-resident, so "global pass" here means "L2 pass," and the kernel is latency/issue-bound, not DRAM-bound.

---

## 1. Memory traffic budget — the numbers the prompt asked for

Counting **full-row traversals** per row, for the medium single-CTA cell (n = 16384, k = 2048):

| | global/L2 full passes | shared full passes | candidate-only passes |
|---|---|---|---|
| **stock / #55314 fast path** | **2** (coarse hist; then collect+emit+stash) | 0 | 4 gathers of ≤4096, *from global by index* |
| **#55314 overflow path** | **2 + up to 9** (per descent level: 1 hist + 1 fill; + terminal stash fill) | 0 | 4 |
| **#55122, cached** (always true on GB10 single-CTA) | **1** | **4** (passes 1–3 + emission) | 0 |
| **#55122, uncached** (`force_single_cta`, or FilteredTopK long rows on a ≥128 KB GPU) | **5** | 0 | 0 |
| **proposed** | **1** | **2** | 1 stash pass |

Multi-CTA (`radix_topk`), per CTA per chunk:

| | global | shared full |
|---|---|---|
| stock / #55314 | 1 (chunk load) | 7 (4 rounds + gt count + gt emit + eq emit) |
| #55122 | 1 | 7 (4 rounds + gt count + **eq count** + scan emission) |
| proposed | 1 | **5** (4 rounds + scan emission) |

**Calibrating the model.** Let a global/L2 full pass = 1.0 unit, a shared full pass = `s`. Stock medium = 2.0.
- Preview (`9384be3950`) = 4 global passes ≈ 4.0 predicted; measured 1.76–1.99x = 3.5–4.0 units. Fit is good.
- #55122 = 1 + 4s; measured 1.14–1.31x = 2.28–2.62 units → **s ≈ 0.27–0.36**.

Two independent points, consistent. Pass count *is* the right predictor **on the single-CTA path**, with shared passes weighted ~⅓.

**But it is the wrong predictor for the 1.54x cell.** At 64 rows × 32768 both stock and #55122 do 1 global + 7 shared per chunk — *identical pass counts*. The regression there is not traffic. It is:
- `cub::BlockScan&lt;uint32_t,1024&gt;` + `__syncthreads()` once per 1024-element tile in the emission (line ~1075): 16 block scans and ~32 block-wide syncs per 16384-element chunk, replacing stock's plain predicated store;
- one extra inter-CTA `red_release`/`wait_ge` round trip;
- the serialized `if (tx == 0)` loop of `ld_acquire` over `det_gt_counts`/`det_eq_counts`.

I would not have found that from the traffic budget, and any design that only optimizes passes will not fix that cell.

---

## 2. The design

Three levers, deliberately separable so each can be A/B'd and reverted alone.

### L3 (do this first — cheapest, biggest single win, cannot change the selected set)

**L3a — blocked, 4-items-per-thread emission.** In `det_select_row`'s emission loop (`persistent_topk.cuh:281–301`) and the identical loop in `radix_topk` (~1063–1090), replace the striped one-element-per-thread tile with a **blocked** `ITEMS_PER_THREAD = 4` layout: thread `t` owns indices `[tile_base + 4t, tile_base + 4t + 4)`, read from `keys[]` / `shared_ordered[]` as one `uint4` (conflict-free 128-bit LDS). Compute a serial 4-element scan in registers into a per-thread aggregate `(gt_count | eq_count &lt;&lt; 16)`, run **one** `ScanT::ExclusiveSum` per 4096 elements, then re-walk the 4 registers adding the local serial offsets. The placement rule is unchanged: `pos = g + min(e, fin)`.

This cuts block scans and block-wide syncs **4×** with no change to the emitted set or order — the position formula is a function of index only. Blocked layout preserves index order within a thread and across threads, which is exactly what the formula needs.

**L3b — derive the multi-CTA `gt`/`eq` counts from the round histograms.** Delete both chunk-scanning count loops in `radix_topk` (lines ~991–1013 and ~1017–1027). In each round `r`, after `thr_r` is known from the global suffix scan and *before* the next round zeroes `local_histogram`, accumulate
`my_gt_local += Σ_{b &gt; thr_r} local_histogram[b]` (a 256-element warp reduction), and in round 3 set `my_eq_local = local_histogram[thr_3]`.
`local_histogram` already counts exactly the prefix-matching local elements, so the accumulation is a partition of the local chunk with no double counting. This removes **2 full shared chunk passes** for the cost of 4 warp reductions over 256 ints.

Expected on the 64×32768 cell: 1.54x → roughly 1.0–1.1x. L3 alone is ~60 lines and is the only lever I would commit to without further measurement.

### L1 — a finer first bucket

`convert_to_uint32_v2(x) &gt;&gt; 24` (byte 3) is `sign | exponent[7:1]`. It drops the exponent's LSB, so **byte 3 is constant across pairs of binary octaves** — it is a catastrophically coarse first split for real logits, which occupy two or three octaves. That is *why* #55122's passes 1 and 2 have to scan the whole row: the threshold bin after pass 0 holds a large fraction of it. Stock avoids this by using an fp16-derived bucket (`convert_to_uint8` = `sign | exp[5] | mant[9:8]`), 4 sub-bins per octave — 8× finer — which is why its stash rarely overflows.

Change `det_select_row` pass 0 to a **4096-bin histogram on `key &gt;&gt; 20`** (`sign | exponent[8] | mantissa[22:20]`, 8 sub-bins per octave over the full fp32 range). 16 KB of shared. Suffix-scan it with one `cub::BlockScan&lt;uint32_t,1024&gt;` over 4 items/thread, same shape as the existing decode-path pair scan.

Do **not** reuse stock's fp16 bucket. It is monotone but not injective at the extremes: everything below ~6e-5 collapses to the zero bin and everything above 65504 to the inf bin, which is precisely what forces #55314's descent. Taking the bits straight out of the ordered fp32 key has no such collapse and no fp16 round-trip.

Residual after L1: 20 bits, resolved in two 10-bit passes (1024 bins, 4 KB each).

### L2 — a candidate stash whose capacity cannot affect correctness

This is the key structural difference from #55314 and from the rejected preview.

In pass P1 (the first shared full pass), for each `i` with `keys[i] &gt;&gt; 20 == thr12`:
```
slot = atomicAdd(&amp;cand_count, 1);          // scratch slot only, order-independent
if (slot &lt; CAND_CAP) cand[slot] = (uint16_t)i;   // n &lt;= 16384 fits uint16
atomicAdd(&amp;hist10a[(keys[i] &gt;&gt; 10) &amp; 0x3FF], 1); // built for ALL survivors, stashed or not
```
Then passes P2 iterate `cand[0..cand_count)` instead of `0..n` — **unless** `cand_count &gt; CAND_CAP`, in which case they fall back to the full shared scan. `cand_count` is a count, so the overflow decision is deterministic; the stash's *contents* only ever feed commutative integer `atomicAdd`s into histograms. **The stash is a pure speed optimization; the selected set never depends on its capacity or on arrival order.** That is the property #55314's stash lacks and the reason its exactness fix needed a whole descent to bolt on.

No `atomicAdd` determines an output slot or which tied element is kept: output positions come only from the `ExclusiveSum` in the emission, and tie choice only from `e &lt; fin`.

### Resulting shape of `det_select_row`

```
P0  global full   load float4 → keys[i] (shared) + hist12[key&gt;&gt;20]   [1 global pass]
    scan hist12 (4096 bins, 4 items/thread) → thr12, above, pop
    early exit if pop == remaining  (already present, line 271)
P1  shared full   stash survivors + hist10a[(key&gt;&gt;10)&amp;0x3FF]         [1 shared pass]
    scan hist10a → thr10a; prefix |= thr10a&lt;&lt;10; early exit
P2  stash only    hist10b[key &amp; 0x3FF]  (full shared scan if overflowed)
    scan hist10b → pivot, fin
P3  shared full   blocked 4-item emission, pos = g + min(e, fin)     [1 shared pass]
```

Shared budget at n = 16384: keys 65536 + hist12 16384 + hist10 4096 + scan tmp ~128 + stash 8192 (4096 × uint16) = **94336 ≤ 101376 − static**. Tight but it fits; `det_select_row_bytes` must be extended to account for the histogram and stash and remain the single source of truth used by both `topk.cu:139` and the kernel.

### Path coverage — explicitly

- **`histogram_2048_topk` (n ≤ 8192)** — already dead on the #55122 branch; the design does not resurrect it. Rows n ≤ 8192 go through the same `det_select_row`, keys in shared (32 KB, comfortable). An optional Tier-1 variant keeps the 8 keys per thread in *registers* instead (8 uint32 at 1024 threads; the existing decode path already keeps 8 uint16 bins there). I would **not** ship that: the register variant's baseline is stock's decode path, which is already ~1 global pass, so it competes at ~1.0–1.15x, not as a win, and 8 more registers against a 64-reg/thread budget at 1024 threads risks spills. Check with `-Xptxas -v` before believing any of it.
- **`histogram_256_topk` (8192 &lt; n ≤ 16384)** — the main target. This is where the 1.14–1.30x lives and where L1+L2+L3a apply in full.
- **`radix_topk` (n &gt; `RADIX_THRESHOLD` = 16384, multi-CTA)** — gets **L3a + L3b only**. Keep the 4×256-bin round structure. Widening its first round to 4096 bins would require `RadixRowState::histogram` to become `[3][4096]` = 48 KB/group × up to 48 groups = 2.3 MB of workspace, a Python-side allocation change, and 16× the cross-CTA histogram atomic traffic — for a path whose cost is dominated by six inter-CTA barriers, not by its shared passes. Not worth it. What the multi-CTA case *does* need beyond the single-CTA design: nothing new. The per-CTA prefix machinery (`det_gt_counts`/`det_eq_counts`, `gt_before`/`eq_before`) that #55122 already added is what makes the position formula work across CTAs, and it stays; L3b only changes where the two counts come from.
- **Uncached fallback** (`force_single_cta`, or `FilteredTopKUnifiedKernel` with rows longer than fit — reachable on H100/B200 with 227 KB optin and long contexts, dead on GB10 since `max_smem_per_block &gt;= 128*1024` is false at 101376). Here P1/P2/P3 read global: **3 global passes** versus #55122's 5. A ~0.6x improvement on the worst case in the file.

---

## 3. Expected cost, and why it does not add full-row passes

Using the calibrated model (global = 1.0, shared = 0.27–0.36, stash ≈ 0.02, block-scan overhead cut 4×):

| cell | stock | #55314 | #55122 | proposed (predicted) |
|---|---|---|---|---|
| n = 16384 single-CTA | 1.00 | 1.00–1.09 | 1.14–1.30 | **0.85–1.00** |
| 64 × 32768 multi-CTA | 1.00 | 1.04 | 1.54 | **0.95–1.10** |
| n ≤ 8192 | 1.00 | 1.00 | (folded into det path) | **1.00–1.15** |
| uncached long row | 1.00 | 1.00 | ~2.5 | **~1.5** |

**The specific reason it adds no full-row passes over #55314:** #55314 spends its two full-row passes on (i) building a coarse histogram and (ii) partitioning the row into "definitely in" + "stash." The proposed design spends the same two on (i) building a *finer* histogram while caching the ordered keys, and (ii) partitioning into "survivor" + "stash" while building the next histogram. It then spends a **third** on the index-ordered emission — but that third pass reads **shared**, weighted ~0.3, because pass (i) cached the keys. Net: 2.0 units → ~1.75 units. The preview at `9384be3950` added the same emission pass but read it from **global** and paid an extra global gt-count pass on top, which is exactly the 2.0 extra units its 1.76–1.99x reflects.

The reason it removes passes versus #55122: #55122's descent passes 1–3 each scan all n because byte 3 of the fp32 key is too coarse to shrink the candidate set. L1 shrinks it (12 bits instead of 8, and no dropped exponent LSB), and L2 then lets passes 2+ touch only the candidates.

---

## 4. Failure modes and the cheapest test for each

| # | failure mode | cheapest test |
|---|---|---|
| 1 | First bucket not monotone in the true key (would corrupt the pivot search) | Host-only unit test: assert `convert_to_uint32_v2` is strictly increasing over a dense float sweep plus ±0, subnormals, ±inf, and that `&gt;&gt; 20` is non-decreasing. Milliseconds, no GPU. |
| 2 | Stash overflow silently truncating (the #55314 defect class) | One shape: n = 16384 with 8192 elements sharing the top 12 bits straddling the boundary, k = 2048. Assert the set equals `torch.topk` value-desc/index-asc, and self-consistent over 6 calls. |
| 3 | `fin` off-by-one at the pivot | Three tiny shapes with *known exact* expected output, no reference kernel: all-equal row → `[0..k-1]`; a row with exactly k above pivot (`fin == 0`); one with k−1 (`fin == 1`). |
| 4 | Blocked emission tail: `ITEMS_PER_THREAD` not dividing n → dropped/duplicated index | n ∈ {4095, 4096, 4097, 16383, 16384, 16385}. Assert output is strictly ascending, unique, length k. No reference needed — this is a self-check. |
| 5 | **L3b: `gt`/`eq` derived from round histograms disagreeing with a direct count** (the riskiest new step) | Debug-only build that *also* runs the old chunk-scan counts and `assert`s equality; run it once over the 81-shape sweep, then delete the assert. Far cheaper than debugging a wrong pivot downstream. |
| 6 | Cross-CTA index-interval assumption broken when the last chunk is empty | `seq_len = ctas_per_group * chunk_size + 1` and `= chunk_size` exactly, so one CTA gets `actual_chunk_size == 0`. Assert ascending output. |
| 7 | Shared-memory sizing drift between `topk.cu` and the kernel → silent overrun into `keys[]` | Keep #55122's discipline: one `constexpr det_select_row_bytes` used by host and device, a `STD_TORCH_CHECK` against `optin − fa.sharedSizeBytes`, plus one `compute-sanitizer --tool memcheck` run over the sweep. |
| 8 | Register spill on any Tier-1 register variant | `-Xptxas -v` at build time; &gt; 64 regs/thread at 1024 threads is a launch failure, not a slowdown. |
| 9 | Reproducibility regression anywhere | The existing 81-shape × 6-call harness is the acceptance gate. Nothing merges below 81/81. |

Two additional bugs I found while reading, worth their own tests regardless of which design wins:

- **#55314 splits ±0 into different buckets.** Its `convert_to_uint32_v2` (pr55314 line 50) has *no* zero canonicalization, and `convert_to_uint8(-0.0f) = 0x7F` while `convert_to_uint8(+0.0f) = 0x80`. On a row containing both zeros near the boundary it orders `+0.0` above `-0.0`, violating "equal values tie-break by index." #55122 fixes this at line 53. Test: a row with interleaved ±0 and k landing inside them.
- On GB10, `max_smem_per_block &gt;= 128*1024` is false (101376), so **`FilteredTopKRaggedTransform` never runs here** — the `num_rows &gt; 32` branch at `topk.cu:37` always falls through to the persistent kernel. The 81/81 result is a GB10 result; it says nothing about the filtered path on a 227 KB GPU. On the #55122 branch that path also calls `det_select_row`, so it is covered by construction — but *uncached* for rows beyond ~24k, at 5 global passes. Nobody has measured that.

---

## 5. What makes this not worth doing — and what I would actually merge

**Be honest about the risk/reward.**

The full design (L1 + L2 + L3) re-introduces, into the one kernel in this file that has already produced two determinism bugs, a candidate stash, an overflow fallback, a second histogram width, and a per-tier shared-memory layout. #55122 is a **net deletion** — it removes both histogram paths (~700 lines) and makes the correctness argument small enough to read in one sitting. Trading that for 10–25% on a kernel whose share of a decode step nobody in this repo has measured is the wrong direction. Before spending days on L1+L2, run `nsys` on one decode step and compute `(ratio − 1) × kernel_time × layers × steps`. If that is under a percent of the step, stop.

**#55314 is not a candidate to "win."** It scores 0/81 self-consistent and 0/81 index-canonical. The bug that motivated this work is that identical requests fork at temperature 0 because the sparse attention sums selected keys in output order; #55314 does not touch output order at all. An exactness fix at 1.0x that leaves the divergence in place buys nothing. Its own convert path also still splits ±0. It is a good *component* — its descent idea is what L1+L2 generalize — but it is not a merge candidate on its own.

**Recommended sequencing:**

1. **Merge #55122 now.** It is the only thing in the file that satisfies the contract, and its cost is bounded (1.14–1.30x on the shapes that matter, 1.54x on one).
2. **Land L3 (L3a + L3b) as a separate perf PR.** ~60 lines, provably cannot change the selected set or its order (the position formula is a pure function of index, pivot and `fin`), and it targets the one cell where #55122 is genuinely bad — a cell the traffic budget does not even explain. Re-measure the 81 shapes.
3. **Only then decide on L1 + L2**, against the post-L3 numbers. If L3 lands 64×32768 near 1.05x and the medium cell stays at ~1.2x, the remaining 20% on one path is probably not worth the stash's complexity, and I would leave it.

One caveat on my own prediction: the 0.85–1.00x for the medium cell rests on `s ≈ 0.27–0.36` fitted from two points on one machine. It is a plausible estimate, not a measurement, and the 4096-bin suffix scan (a `BlockScan` over 4 items/thread plus 16 KB of histogram zeroing per row) is a real fixed cost I have not priced. If the medium cell comes back at 1.05x instead of 0.9x, that is the reason, and L1 should then be dropped in favour of L2 alone on the existing 256-bin structure.