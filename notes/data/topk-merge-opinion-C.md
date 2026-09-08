I read both versions of the kernel, the launcher, and the test suite, and I measured the two facts my cost argument depends on directly on the box.

## 0. Facts established on this machine (not estimates)

**GB10 / sm_121 device properties** (compiled and ran a `cudaGetDeviceProperties` probe):

```
sms=48  sharedMemPerBlockOptin=101376  sharedMemPerSM=102400
regsPerSM=65536  maxThreadsPerMultiProcessor=1536
```

`maxThreadsPerMultiProcessor = 1536` is load-bearing: with `kThreadsPerBlock = 1024`, **two blocks per SM are impossible on GB10**. ptxas confirms it for both branches:

```
ptxas warning : Value of threads per SM for entry ..._topk_kernel... is out of range.
                .minnctapersm will be ignored
ptxas info    : Used 64 registers, used 1 barriers, 4256 bytes smem   (#55122)
ptxas info    : Used 64 registers, ... 32 bytes cumulative stack size (#55314)
```

Consequences: (a) `__launch_bounds__(kThreadsPerBlock, 2)` is inert here, both branches already get 64 registers and **zero spills** — `reg_bins[8]` in `histogram_2048_topk` really does stay in registers; (b) occupancy is 1 CTA/SM regardless, so **shared memory up to ~99 KB is free**, and (c) with no co-resident block, every `__syncthreads()` is *exposed* latency. That third point is what actually explains #55122's cost, see §2.

**Structural facts from reading the code:**

- `pr55314:persistent_topk.cuh:36` has `RADIX_THRESHOLD = 32768`; the checked-out #55122 branch has `RADIX_THRESHOLD = 16384`. So at 64 rows × 32768, #55314 runs **one CTA** through `histogram_256_topk`, while #55122 runs a **2-CTA cooperative launch** with 6 inter-CTA barriers and a 64 KB chunk. That is a *routing* change, not a rescanning cost.
- Why it was lowered: `det_select_row` caches the **full 32-bit** ordered key (`det_select_row_bytes = fixed + n*4`). 16384 × 4 = 64 KB fits under 101376; 32768 × 4 = 128 KB does not.
- In #55122, `histogram_2048_topk` and `histogram_256_topk` are **dead code** — grep shows no call sites; every single-CTA row goes to `det_select_row`. So #55122's 1.14–1.30 x at n = 16384 is partly the loss of the 11-bit (2048-bin) first split, replaced by an 8-bit one.
- #55122's multi-CTA `radix_topk` Stage 3 does **the same three chunk passes** as stock (stock: gt-count, gt-emit, eq-emit; #55122: gt-count, eq-count, packed-scan emit), plus one barrier and a 64-entry table read by thread 0. Its multi-CTA path is therefore ≈ stock cost. This corroborates that the 1.54 x is the routing change.

**A defect in #55314 that any merged design must fix:** `convert_to_uint8` and `decode_bin` do *not* canonicalise signed zero (only `convert_to_uint32_v2` does, and only on the #55122 branch). `decode_bin(+0.0) = 0x400`, `decode_bin(-0.0) = 0x3FF` — the coarse histogram ranks every `+0.0` above every `-0.0`, so the tie-by-index rule is already broken before any emission logic runs. `test_persistent_topk_signed_zero_ties_by_index` exists on `fix/persistent-topk-deterministic` and is **absent from `pr55314`** (verified by grep on the PR's own test file).

---

## 1. Candidates

**C1 — Post-hoc sort of the TopK output on top of #55314. Rejected.** It fixes order but not the set: #55314 still picks ties by arrival in two places — `atomicAdd(&amp;shared_final_k, -1)` in the last refine pass, and the `bp &lt; MAX_BUFFERED_ITEMS` / `bp &lt; DBUF` stash clip that is *reachable* at `level == 4` (the `pop &lt;= DBUF || level == 4` branch), i.e. exactly when there are more exact fp32 ties than stash slots. `test_persistent_topk_pivot_ties[num_ties=16384]` hits it. And the sort itself is wrong-shaped: a bitonic sort of 2048 keys in a 1024-thread block is 66 barrier-separated stages, and on GB10 (1 CTA/SM) every barrier is exposed latency. It would cost more than the extra data pass it saves.

**C2 — Dense n-sized rank array (`int32 rank[n]`) written by scatter, then compacted. Rejected.** Correct, but it is the bitmap idea at 32× the memory and 32× the scan width. Superseded by C5.

**C3 — Explicit A/B merge (coarse-greater set compacted into `Aidx[a]` during the stash pass; `g` = number of coarse-greater elements before each stash entry packed into the stash word; final position `g+b` for tie survivors and `a+c(a)` for the definite set, with `c` from a `cnt[]` histogram over `g` values). Rejected on review risk, not on cost.** It also adds no full-row pass, but it needs `(g&lt;&lt;15)|i` packing that runs out of bits on the multi-CTA path (index to 163840 = 18 bits + g to 2048 = 12 bits), an extra `cnt` histogram plus scan, and three interlocking rank formulas. C5 gets identical asymptotics with *one* formula, and that formula is the one #55122 already ships and tests.

**C4 — Fix #55122 in place (recommended, see §3).** Four changes, no new algorithm.

**C5 — Selection bitmap + one word-level ranked emission (the design, §2).** The ordering is obtained *by construction from the traversal order of a bit-packed image of the selection*, exactly as the prompt suggests: no index is ever compared, the bitmap **is** the index-ordered buffer, and the rank of a selected element is a prefix over bit positions.

---

## 2. The design: selection bitmap + word-level ranked emission

### 2.1 Idea

Every emission site in #55314 currently answers "which output slot do I get?" with an `atomicAdd`. Replace all of them with "set my bit", which is order-independent, and compute all slots once at the end from the bitmap. Because a bitmap holds 32 elements per word, ranking it is **1/32 of a row pass**, so the ordering property costs no full-row pass anywhere.

Two shared bitmaps of `W = ceil(n/32)` words each:

- `sel[]` — elements known to be in the answer (strictly above the final pivot).
- `tie[]` — elements exactly equal to the final pivot; only the lowest `fin` of them by index are kept.

`sel` and `tie` are disjoint by construction. Keep #55314's existing `shared_output_count` / `shared_final_k` scalars, but use them only as **counts** (an `atomicAdd` that produces a count and never a slot is order-independent and permitted); `fin = TopK - |sel|`.

### 2.2 The emission primitive (new, ~40 lines)

```cuda
__device__ __forceinline__ void mark(uint32_t* bmp, int i) {
  atomicOr(&amp;bmp[i &gt;&gt; 5], 1u &lt;&lt; (i &amp; 31));
}

template &lt;int N_THREADS&gt;
__device__ void emit_from_bitmap(const uint32_t* sel, const uint32_t* tie,
                                 int words, uint32_t fin, int base_index,
                                 uint32_t run_sel, uint32_t run_tie,
                                 int32_t* out, int top_k, void* scan_tmp);
```

Per tile of `N_THREADS` words, one packed `cub::BlockScan::ExclusiveSum` of `popc(sw) | (popc(tw) &lt;&lt; 16)` gives `s_before` and `t_before`. For a bit `j` in word `w`, with `m = (1u&lt;&lt;j)-1`:

```
A = s_before + __popc(sw &amp; m)                  // definite elements before it
B = min(fin, t_before + __popc(tw &amp; m))        // kept ties before it
emit iff (sw&gt;&gt;j)&amp;1  ||  ((tw&gt;&gt;j)&amp;1 &amp;&amp; t_before + __popc(tw &amp; m) &lt; fin)
pos = A + B
out[pos] = base_index + 32*w + j
```

This is **literally #55122's `pos = g + min(e, fin)`**, evaluated once per 32 elements instead of once per element. The per-word loop iterates only over set bits (`__ffs` / `m &amp;= m-1`), so total work across the block is `TopK` iterations, ~2 per thread.

At `n ≤ 32768`, `W ≤ 1024` = **one tile**: one BlockScan, one `__syncthreads()`. #55122's emission at n = 32768 would be 32 tiles = 32 barriers.

### 2.3 Path 1 — `histogram_2048_topk` (n ≤ 8192)

Shared memory: `W = 256`, so `sel` + `tie` = 2048 bytes. The current layout ends at `SBASE + 8 = 8192 ints = 32768 B` and the launcher already guarantees `kSmemMedium = 35968 B`, so the bitmaps fit in **existing slack** — zero extra allocation.

Edits (line numbers from `git show pr55314:csrc/libtorch_stable/persistent_topk.cuh`):

- Zero `sel`/`tie` alongside the `decode_smem[SBASE + tx] = 0` init (~line 186).
- Replace every `pos = atomicAdd(&amp;decode_smem[sOUT_abs], 1); if (pos&lt;TopK) output_indices[pos]=idx;` with `mark(sel, idx); atomicAdd(&amp;decode_smem[sOUT_abs],1);` — five sites: the `above_mask` ballot block (~316), the overflow "definite members" fill (~356), the per-level "definite members" fill (~397), the `remaining_k == 0` break (~478), and the `bin &gt; ref_thr` branch in the refine loop (~500).
- Replace the tie clip `slot = atomicAdd(&amp;decode_smem[SBASE+sFIN], -1); output_indices[TopK-slot] = idx;` (pass 3, ~505) with `mark(tie, idx)`.
- Replace the "all buffered fit" early-out (`raw_buf0 &lt;= remaining_k`, ~772) with `mark(sel, bufs[0][i])`.
- **At `level == 4` with `pop &gt; DBUF`, do not stash at all** — `mark(tie, idx)` for every participant and set `fin = rem`. The bitmap has no capacity, so the `bp &lt; DBUF` / `bp &lt; MAX_BUFFERED_ITEMS` clip stops deciding anything. This is where exactness stops depending on a buffer size.
- Turn the four early `return`s into a single exit that calls `emit_from_bitmap&lt;1024&gt;(sel, tie, 256, fin, 0, 0, 0, output_indices, TopK, scan_storage)`.
- Fix `decode_bin`: canonicalise `±0` (`if ((bits &amp; 0x7FFF) == 0) bits = 0;`) before the order-preserving flip.

### 2.4 Path 2 — `histogram_256_topk` (8192 &lt; n ≤ 32768)

Identical edits; the emission sites are at ~635 (`bin &gt; threshold_bin` fast path), ~684/~712/~730 (overflow levels), and the `pass == 3` tie clip at ~836. `W ≤ 1024`, so the bitmaps are 8 KB. Raise `kSmemMedium` by `2 * (RADIX_THRESHOLD/32) * 4 = 8192` → 44160 B and take `max()` in the launcher. On GB10 that is free (1 CTA/SM either way); on a 2048-thread/SM part 44 KB is still under the 50 KB that would cost the second block. Fix `convert_to_uint8` for `±0` as above.

### 2.5 Path 3 — multi-CTA `radix_topk` (n &gt; 32768)

The four cooperative radix rounds stay exactly as they are: the global histograms are summed with `atomicAdd`, which is order-independent, so `ordered_pivot` is already deterministic in stock. Only Stage 3 changes.

Each CTA keeps **per-CTA shared bitmaps over its own chunk** (`chunk/32 ≤ 24576/32 = 768` words = 3 KB each; no global bitmap, no workspace change):

1. One chunk pass: `shared_ordered[i] &gt; pivot → mark(sel,i)`, `== pivot → mark(tie,i)`. This *replaces* stock's three chunk passes and #55122's three, so this path gets **cheaper than stock**.
2. `__popc`-reduce both bitmaps, publish the two totals into `RadixRowState::det_gt_counts / det_eq_counts` (already added by #55122). One barrier.
3. Each CTA sums the lower CTAs' entries → `run_sel = sel_before`, `run_tie = tie_before`, `fin = TopK − Σ sel_count`.
4. `emit_from_bitmap&lt;1024&gt;(..., base_index = my_chunk_start, run_sel, run_tie, ...)`.

CTA `c` owns a strictly lower index interval than `c+1`, so CTA prefix + intra-CTA bit rank *is* the global ascending rank. No atomic determines a slot; no sort.

**What the multi-CTA case needs that the single-CTA case doesn't:** (a) the two per-CTA count publications and the extra barrier — already present in #55122; (b) `fin` must be computed from the *global* `sel` total, not the local one; and (c) **`chunk_size` must be a multiple of 32**, or a bitmap word straddles two CTAs and two CTAs race on it. Add to `launch_persistent_topk` in `topk.cu` (the existing `chunk_size = ((chunk_size + vec_size - 1)/vec_size)*vec_size` line): round to `lcm(vec_size, 32) = 32`, plus `STD_TORCH_CHECK(chunk_size % 32 == 0)`.

### 2.6 Path 4 — `FilteredTopKUnifiedKernel` (num_rows &gt; 32)

Same treatment; its slot source is `s_counter`. Not in the required list but it is the path `launch_persistent_topk` takes for `num_rows &gt; 32`, and leaving it arrival-ordered would leave the op non-reproducible for exactly the batch sizes that matter.

---

## 3. Cost, and why it adds no full-row pass

Per row, relative to #55314:

| added | cost |
|---|---|
| zero two bitmaps | `n/16` word writes ≈ 1/64 of a row pass |
| `atomicOr` per selected/tie element | replaces one `atomicAdd` + one global store; contention is *lower* (n/32 words vs. one counter) |
| finalize | `ceil(n/32)` shared word loads + 1 BlockScan + 1 barrier + `TopK` global stores |
| removed | all `if (pos &lt; TopK)` guards, all stash capacity clipping |

**The specific reason no full-row pass is added:** the output order is derived from a bit-packed image of the selection, not from the data. The prefix scan that produces the ranks runs over `n/32` words, and the tie-break rule ("lowest indices win") is a prefix property of that same bitmap — `min(fin, #tie-bits-before)` — so it needs neither a comparison of indices nor a second visit to the row. The only structure that would force an extra full-row pass is one that re-derives per-element `&gt;`/`==` flags at emission time; the bitmap caches those flags at 1 bit each, at the moment they were already computed.

**Expected:** 1.00–1.05 x of #55314 on the fast paths; at or below *stock* on the multi-CTA path (3 chunk passes → 1 chunk pass + a word scan).

**Where it does not help.** #55314's descend/overflow branch still does up to two full-row **global** rescans per level (`key_participates` re-reads `logits[idx]`, in two loops per level, up to 4 levels, plus a stash loop) — up to ~9 row reads. That branch is not rare for QSA: `test_persistent_topk_narrow_value_range` (values within 1e-3 of 1.0, "as a trained indexer head produces", issue #51782) puts the entire row in one coarse bin on both the 2048-bin and the 256-bin paths. Orthogonal fix, worth measuring separately: cache the row's ordered keys once in shared (full `uint32` for n ≤ 24576, or the **top 16 bits** for n ≤ 32768 at 2 bytes/element = 64 KB) and make every descend level a shared-memory pass.

---

## 4. Failure modes and the cheapest test for each

| failure | cheapest catch |
|---|---|
| `chunk_size` not a multiple of 32 → two CTAs race on one bitmap word | a host-side `STD_TORCH_CHECK(chunk_size % 32 == 0)` in `topk.cu` (turns a data race into a launch error); as a test, add `seq_len=49157` to `test_persistent_topk_deterministic` |
| bitmap not cleared between rows (groups are reused in the persistent loop) | existing `test_persistent_topk_reused_group_after_short_row` and `test_persistent_topk_degenerate_lengths` (row length 0 in a 64-row batch) |
| `±0.0` split across coarse bins | existing `test_persistent_topk_signed_zero_ties_by_index` — **it is on the `fix/persistent-topk-deterministic` branch and not on `pr55314`; #55314 fails it as written** |
| `\|sel\| + min(\|tie\|, fin) ≠ TopK` → `-1` holes or writes past `TopK` | existing `test_persistent_topk_pivot_ties[2047,2048,2049,4096,4097,16384,16385]` — that is exactly the `fin` boundary sweep; plus a debug-only in-kernel assert that the emitted count equals `TopK` |
| the "keep the lowest `r` set bits of a word" loop, at `r == 0` and `r ≥ popc(tw)` | existing `test_persistent_topk_all_equal` (`fin = TopK`, every bit a tie) and `test_persistent_topk_exact_bin_boundary` (`fin = 0`) bracket both ends |
| overflow/descend branch left inconsistent after removing the tie clip | existing `test_persistent_topk_narrow_value_range`; add `seq_len=32768` to its matrix |
| shared memory over the optin after adding the bitmaps | `static_assert` on the layout + run the matrix at `num_rows=64, top_k=2048, seq_len=40000` (largest chunk request) |
| reproducibility itself | the existing 6-call loop in `test_persistent_topk_deterministic`; add `kind="clustered"` (`round(x*8)/8`), which is where the arrival-order tie-break actually bites |

---

## 5. What I would actually do, and why the merged design may not be worth building

**Do not merge #55314 as the answer.** 0/81 self-consistency means it does not fix the bug that motivated the work — greedy decoding forking between identical requests — and its own tie clips leave a scheduling-dependent *set* in precisely the clustered case its exactness fix targets.

**Merge #55122, then fix it in place (candidate C4).** I believe most of the measured penalty is not the rescanning:

1. **The 1.54 x at 64 × 32768 is a routing artifact, not a rescan cost.** `RADIX_THRESHOLD` was lowered 32768 → 16384 only because `det_select_row` caches a 4-byte key and 32768 × 4 = 128 KB &gt; 101376. Cache the **top 16 bits** instead (2 bytes → 64 KB covers 32768), do the 11-bit coarse split on those bits, and gather the full `fp32` key from global for the ≤ few-thousand tie candidates only. Restore `RADIX_THRESHOLD = 32768`. ~20 lines. My prediction: 1.54 x → ~1.05 x, because the 2-CTA cooperative launch with 6 barriers simply stops happening at that shape.
2. **Emission with `ITEMS_PER_THREAD = 8`.** `det_select_row`'s emission loop is one `BlockScan` + one `__syncthreads()` per **1024** elements — 16 barriers at n = 16384, 32 at 32768. With 1 CTA/SM on GB10 (measured) nothing hides them. Eight items per thread → 2 barriers.
3. **Compact the threshold-bin candidates after pass 0**, so passes 1–3 histogram over the candidate list instead of filter-scanning all `n` cached keys. The compaction may use `atomicAdd` — it determines only membership of an order-independent set and is never clipped. This takes `det_select_row` from 4 full passes over `n` to 2 (one global, one shared) — the same count as #55314.
4. **Restore the 11-bit first split** (2048 bins on the top bits of the ordered key, 8 KB — free at 1 CTA/SM). #55122 deleted `histogram_2048_topk` and dropped to an 8-bit first split; the finer split cuts the threshold-bin population ~8×, which is what makes (3) cheap, and cuts shared-atomic contention 8×.

Then re-measure. If (1)–(4) land near 1.05 x, **the bitmap design in §2 is not worth building**: it touches four kernels (`histogram_2048_topk`, `histogram_256_topk`, `radix_topk`, `FilteredTopKUnifiedKernel`) plus the launcher, in a file two PRs are already contending over, and it *re-introduces* the two code paths that #55122 deletes. Fewer paths that can each be nondeterministic in their own way is a real correctness argument, and it is the one #55122 makes.

Build §2 only if step 2 above leaves a visible cost — its unique property is that it removes the last full-row pass entirely, and it is the only design here that also removes stash-capacity clipping as a correctness concern.

**One number nobody has produced, and it decides everything:** what fraction of the QSA indexer step `persistent_topk` actually is on GB10. If it is ≤ 3 %, then even #55122's un-optimised 1.30 x buys back ≤ 0.9 % end to end, and the correct action is to merge #55122 unchanged, close #55314 (folding in its `±0` coarse-bin fix and its exactness descend for the narrow-range case), and spend the effort elsewhere. I would measure that before writing any of the code above.