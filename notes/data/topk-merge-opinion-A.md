I read all three versions on the box (stock `29af8bd672`, `pr55314`, and the checked-out `fix/persistent-topk-deterministic` = #55122), plus the launcher diff, and measured the GB10 device limits and the compiled register/smem footprint of each. Findings and the proposal follow.

---

# 0. Machine facts that constrain every option

Measured on this box (`cudaGetDeviceProperties`, sm_121):

```
NVIDIA GB10  cc=12.1  SMs=48
sharedMemPerBlockOptin  = 101376   sharedMemPerMultiprocessor = 102400
maxThreadsPerSM         = 1536     regsPerSM = 65536
```

`ptxas -arch=sm_121 -v` on `persistent_topk_kernel&lt;2048,4&gt;`:

| | registers | spills | static smem |
|---|---|---|---|
| stock | 63 | 0 | 0 |
| #55314 | 64 | 0 | 0 |
| #55122 | 64 | 0 | 4256 (its `__shared__ det_scan_tmp`) |

Four consequences that decide the design:

1. **1024 threads/CTA + 1536 threads/SM ⇒ exactly one CTA per SM.** `__launch_bounds__(kThreadsPerBlock, 2)` is unsatisfiable here and ptxas ignores the `2` (all three land at the 64-reg ceiling, 65536/1024). So **~97 KB of dynamic shared memory is free per CTA at zero occupancy cost**, and there are **zero spare registers**. Any design that wants per-element state must put it in smem, not registers.
2. `max_smem_per_block = 101376 &lt; 128*1024`, so the `num_rows &gt; 32 &amp;&amp; max_smem_per_block &gt;= 128*1024` branch in `launch_persistent_topk` (`csrc/libtorch_stable/topk.cu`) is **unreachable on GB10**. Every row here goes through `persistent_topk_kernel`. `FilteredTopKUnifiedKernel` is dead code on this part (but not upstream).
3. #55122's `det_select_row` caches the row only while `fixed + 4n ≤ smem_bytes`; with `fixed ≈ 6.3 KB` and a 97 KB cap that is **n ≲ 22–23k**. That is why #55122 lowered `RADIX_THRESHOLD` from **32768 → 16384**.
4. Therefore **the 1.54x cell at 64×32768 is mostly a path change, not the cost of determinism.** At n=32768 stock/#55314 run single-CTA `histogram_256_topk`; #55122 runs the multi-CTA cooperative `radix_topk`. Its actual determinism delta on the multi-CTA path is one extra smem counting pass, one barrier, and a `BlockScan` emission replacing an `atomicAdd` emission — that is not 54%.

---

# 1. Where the 14–30% actually comes from

This is the load-bearing observation, and it is not in the task description.

Stock's first cut is an **fp16-derived bucket**:
- `decode_bin` (stock.cuh:151) → 11 bits = sign + 5 exp + **5 mantissa** (2048 bins)
- `convert_to_uint8` (stock.cuh:55) → 8 bits = sign + 5 exp + **2 mantissa**

#55122's `det_select_row` throws that away and starts its radix at `key &gt;&gt; 24` of the fp32 ordered key = sign + **exponent[7:1], zero mantissa bits**. QSA indexer scores live in a narrow dynamic range, so essentially the whole row falls into one or two byte-3 bins, the early exit (`bin_pop == remaining`) never fires, and **all four passes run over all n elements, every row.** Add a 5th full-row pass for the emission. That is the 1.14–1.30x, and it is structural, not incidental.

So the merged design is not "pick a compromise". It is: **keep #55314's fp16 coarse bucket and its stash for finding the pivot; keep #55122's rank-at-emission for producing the order; glue them with two shared-memory bitmaps so the emission never re-reads the row.**

---

# 2. The design

### 2.1 The invariant it rests on

`coarse(x) = ordered_key16(__float2half_rn(x)) &gt;&gt; s` is **monotone non-decreasing** in the fp32 value: `__float2half_rn` is monotone (round-to-nearest is), the fp16 ordered-key transform is monotone, `&gt;&gt;` is monotone. Hence, with `P` the exact fp32 pivot (which lies in coarse bin `thr`):

```
coarse(x) &gt;  thr  ⇒  key(x) &gt;  P      (definitely selected)
coarse(x) &lt;  thr  ⇒  key(x) &lt;  P      (definitely rejected)
coarse(x) == thr  ⇒  needs the fp32 compare
key(x) == P       ⇒  coarse(x) == thr (ties can only live in the threshold bin)
```

**This is why no extra full-row pass is needed**: the emission's classification for every element outside the threshold bin is already implied by the coarse pass, and inside the bin it is implied by the refinement — neither needs the float re-read, only a place to record the answer.

**Required fix for the invariant to hold under #55122's ±0 tie semantics** (`convert_to_uint32_v2`, current branch line 50, canonicalises `±0 → +0`): `decode_bin` and `convert_to_uint8` must canonicalise too. Today `-0.0f` → fp16 `0x8000` → key `0x7FFF`, `+0.0f` → key `0x8000` — **different coarse bins for values whose canonical fp32 keys are equal**, which breaks monotonicity exactly at the pivot. One line in each of stock.cuh:55 and :151:

```cuda
uint32_t b = __float_as_uint(x); if ((b &amp; 0x7FFFFFFFu) == 0u) x = 0.0f;
```

(#55314 does not do this and does not canonicalise in `convert_to_uint32_v2` either, so it inherits stock's ±0 splitting.)

### 2.2 New shared state: two bitmaps

```cuda
sel_gt[w]  // bit i set  &lt;=&gt;  key_i &gt;  P
sel_eq[w]  // bit i set  &lt;=&gt;  key_i == P
```

`2 * ceil(n/32)` words, zeroed once per row, written **only by `atomicOr`** — commutative, idempotent, and each element's bit is written by exactly one thread anyway, so the final bitmap is a function of the *set* of writes, never their order. This is the "atomics for scratch are fine" carve-out, used strictly.

**Decode path (n ≤ 8192): costs nothing.** `histogram_2048_topk` uses `decode_smem[0 .. SBASE+8)` = 8192 ints = 32768 B out of `kSmemMedium` = 35968 B — **3200 B are already spare**, and the bitmaps need 2×1024 = 2048 B. Place them at `decode_smem + 8192` and `+ 8448`; they alias nothing (the `histo` / `bufs[0]` aliasing at `BOFF`, noted in #55314's own comment, stays below 8184).

**Medium path (8192 &lt; n ≤ 32768): 8192 B.** Raise `kSmemMedium` from 35968 → 44160. Free on GB10 per fact (1). If a device ever cares, the fallback is `MAX_BUFFERED_ITEMS 4096 → 3072`, which lands on exactly 35968 again at the cost of a higher descent rate.

### 2.3 The emission primitive (new, `namespace vllm::persistent`)

```cuda
template &lt;int N_THREADS, int TopK&gt;
__device__ void emit_ranked_from_masks(
    const uint32_t* sel_gt, const uint32_t* sel_eq,
    int n, uint32_t fin, int32_t* out,
    typename cub::BlockScan&lt;uint32_t,N_THREADS&gt;::TempStorage&amp; tmp);
```

Tile = `4*N_THREADS` elements; thread `t` owns `[4t, 4t+4)` — the same ownership stock's Phase 2 loop already uses, so index order and thread order agree. Each thread reads one nibble from each mask, forms `popc(gt_nib) | (popc(eq_nib) &lt;&lt; 16)`, one `ExclusiveSum`, then walks its own 4 sub-elements in order:

```
g = run_gt + rgt + (gt bits before sub)
e = run_eq + req + (eq bits before sub)
if (gt || (eq &amp;&amp; e &lt; fin)) out[g + min(e, fin)] = i;
```

`run_gt/run_eq` are **32-bit** carries across tiles (only the per-tile totals are packed into 16 bits — `static_assert(4*N_THREADS &lt;= 0xFFFF)`).

This is #55122's `pos = g + min(e, fin)` formula verbatim; the only change is that the flags come from `n/4` bytes of smem instead of `4n` bytes of global, and that **4 elements per thread per scan** cuts the scan count 4× versus #55122's one-element-per-thread emission (8 `BlockScan`s at n=32768; 2 at n=8192). TempStorage aliases `bufs[0]` / `buffered_indices` — the stash is dead by then — so **no new static `__shared__`**, which matters because #55122's 4256 B of statics come straight off `dyn_cap`.

### 2.4 Edits, path by path

**`histogram_2048_topk`** (stock.cuh:160)
| where | change |
|---|---|
| top, next to `if (tx&lt;8) decode_smem[SBASE+tx]=0` | zero `sel_gt`/`sel_eq` (`words = (n+31)/32`) + `__syncthreads()` |
| Phase 2, :248–301 | keep the `reg_bins` walk and the `bin == uthr` stash (scratch). Replace the warp-aggregated `atomicAdd(&amp;decode_smem[sOUT_abs],…)` + `output_indices[…] = elem_idx` for `bin &gt; uthr` with a `sel_gt` bit set (warp-combine the 4 nibbles per lane-octet into whole words; or plain `atomicOr` first, optimise later). Keep the count as a count. |
| #55314's `bin_pop &gt; DBUF` descent | keep verbatim, but its three `atomicAdd(&amp;sOUT)`+write emissions become `sel_gt` sets, and its terminal stash becomes a `sel_eq` set over the full row. **This deletes #55314's DBUF clip and its residual `level == 4` inexactness** — a tie set of any multiplicity now needs zero storage. |
| `raw_buf0 &lt;= remaining_k` shortcut, :307–316 | becomes "set `sel_gt` over the stash, `fin = 0`" |
| Phase 3 refine loop, :345–410 | `bin &gt; ref_thr` → `sel_gt`; at `pass == 3`, `bin == ref_thr` → `sel_eq` **for all of them**. Delete `shared_final_k`'s `atomicAdd(-1)` / `output_indices[TopK-slot]` — that is the tie-order leak. `fin = remaining_k` at loop exit. |
| add | #55122's `bin_pop == remaining` early exit (`fin = 0`, skip dead passes) — free |
| end | `emit_ranked_from_masks&lt;1024,TopK&gt;(...)` |

**`histogram_256_topk`** (stock.cuh:425): the identical transformation. Its collection pass already re-reads `logits` (that path reads the row twice in stock); we do not add a third read because the emission reads bitmaps.

**`radix_topk` / `RadixRowState` / `persistent_topk_kernel`** (stock.cuh:657, :118, :862): **take #55122's version unchanged.** `det_gt_counts[64]` / `det_eq_counts[64]`, `kDetMaxCtasPerGroup`, the single-thread ordered scan of the CTA table after a barrier (order-independent: a sum over a fixed set), and the packed-`BlockScan` emission with `gt_before`/`eq_before` folded in. Drop `output_counter`. **No bitmaps here** — the chunk's ordered keys are already resident in `shared_ordered`, so the classification is free. What the multi-CTA case needs, explicitly: (a) the per-CTA count table + one extra barrier, (b) chunk ownership must stay index-monotone in `cta_in_group` (it is: `my_chunk_start = cta_in_group * chunk_size`), (c) the launcher's `ctas_per_group &lt;= kDetMaxCtasPerGroup` check, (d) `cudaMemsetAsync` zeroing of `row_states` host-side (#55122 already moved it there to fix a real race against CTA-1's first `red_release`).

**`launch_persistent_topk`** (`topk.cu`): keep #55122's launcher fixes — per-call `get_device_prop()`, subtracting `fa.sharedSizeBytes` from the chunk budget, scheduling from `active_width` not the padded pitch, the deterministic low-smem fallback. **Revert `RADIX_THRESHOLD` to 32768** and drop `det_smem_bytes`; the single-CTA fallback becomes `histogram_256_topk`, which has no row-width smem dependence.

**`FilteredTopKUnifiedKernel`, `cooperative_topk.cuh`, `topk_histogram_4096.cuh`**: structurally identical to `histogram_256_topk`; same transformation. Unreachable on GB10 (fact 2) but required for a mergeable PR — this roughly doubles the diff. For rows too wide for the bitmap budget, the emission falls back to re-reading `logits` (one extra full-row pass): still exact, still reproducible, just slower.

---

# 3. Expected cost, and the specific reason

Versus #55314 (≈ stock in the common case):

| path | added work per row | estimate |
|---|---|---|
| `histogram_2048` (n ≤ 8192) | zero 2 KB; `atomicOr` bit sets replacing warp-aggregated `atomicAdd` emission (a wash); **one bitmap walk: 512 smem word loads + 2 `BlockScan`s** | **+2–4%** |
| `histogram_256` (n ≤ 32768) | same, **8 KB bitmaps, 2048 word loads + 8 `BlockScan`s**; output stores become ascending (better coalescing than stock's scattered atomic slots) | **+3–6%** |
| `radix_topk` (n &gt; 32768) | #55122's: one extra smem counting pass over the chunk, one barrier, ≤128 acquire loads by thread 0, `BlockScan` emission replacing atomic emission. Stock also did 3 passes over `shared_ordered` (gt-count, gt-emit, eq-emit); this does 3 (gt-count, eq-count, emit). | **+5–10%** — and the 1.54x cell at 64×32768 disappears entirely, because `RADIX_THRESHOLD` goes back to 32768 and the row stays single-CTA. |

**The specific reason it adds no full-row pass:** the emission's per-element `(gt, eq)` classification is *implied* by work the kernel already does — the coarse pass fixes it for every element outside the threshold bin (monotonicity, §2.1), the refinement fixes it for every element inside it — so the only thing missing is a place to write it down. Two bitmaps are `n/4` bytes of shared memory. The emission therefore reads `n/4` bytes of smem instead of `4n` bytes of global. At n=32768: 8 KB of smem versus the 128 KB global re-read that each of #55122's uncached radix passes performs.

**Where it does add passes:** the `bin_pop &gt; stash` descent. That is data-dependent, it is exactly the case #55314 already pays for, and it costs up to 4 extra full-row global passes. See §5.

---

# 4. Failure modes and the cheapest test for each

| # | failure mode | cheapest test that catches it |
|---|---|---|
| 1 | **Coarse bucket not monotone in the fp32 order** — the design's premise. A future "optimisation" of `decode_bin` breaks everything silently. | Host-only, no GPU: for each of the 65536 fp16 bit patterns compute the fp32 rounding interval that maps to it and assert the intervals are ordered. 65536 iterations, exhaustive, seconds. Pin it with a comment on both bucket functions. |
| 2 | **±0 handled inconsistently** between the coarse bucket and the fp32 key | Row of alternating `+0.0/-0.0`, `k &lt; n`; expect exactly `[0..k-1]`. (#55122 already ships `test_persistent_topk_signed_zero_ties_by_index`.) |
| 3 | **Threshold-bin overflow** (`bin_pop &gt; DBUF`) — the descent is the least-exercised code | Values in a range narrower than one fp16 ULP (e.g. `1.0f + i*1e-9f`) so &gt;4096 share a bucket, `k=2048`. #55314 already has `_clusters_separating_at_bit_depth`; reuse it. |
| 4 | **Tie multiplicity beyond any buffer** (where #55314 still clips) | `n=8192` all-identical, `k=2048` → expect exactly `[0..2047]`. Plus #55122's `test_persistent_topk_pivot_ties` at `num_ties ∈ {2047,2048,2049,4096,4097,16384,16385}`. |
| 5 | **`fin` off-by-one at an exact bin boundary** (`above_coarse == k`, `fin = 0`) | #55122's `test_persistent_topk_exact_bin_boundary`: k values in high bins, remainder strictly lower. |
| 6 | **Cross-row bitmap staleness** — the persistent kernel loops rows per CTA, so a bitmap zeroed outside the per-row barrier interval poisons the next row. Highest-probability new bug in this design. | Force it: 48 SMs ⇒ ~48 CTAs, so run `num_rows=192` with lengths cycling `{1000, 8000, 30000}` and compare against the reference. Each CTA then handles ≥4 rows of *different* lengths and paths. One test, catches the whole class. |
| 7 | **Emission writes past `TopK`** / fewer than k emitted | Keep #55314's `-1` prefill of `output_indices` and assert `(out &gt;= 0).all()` plus `out.shape[-1]` uniqueness. Free rider on every existing test. |
| 8 | **Multi-CTA geometry dependence** — output must not depend on `ctas_per_group` | Same logical row at two paddings that change `ctas_per_group` (e.g. `stride` 40960 vs 65536 with the same `lengths`), assert bit-identical output. Plus `n ∈ {32767, 32768, 32769}` for the threshold seam. |
| 9 | **Tail handling** in the 4-elements-per-thread scan | `seq_len ∈ {8191, 8193, 32767, 32769}` — cheap parametrize on the existing determinism test. |
| 10 | **Reproducibility under scheduling variation** (the original bug) | The existing 6-call self-consistency loop, but additionally with a competing stream running a large GEMM so CTA arrival order actually changes. Cheap, and it is the only test that would have caught the *original* defect at low occupancy. |

---

# 5. Why this may not be worth doing — read this before writing any code

**5.1 The headline number is a median, not a tail.** #55314 and this design both fall into the multi-pass descent when the fp16 coarse bin overflows the stash. The QSA indexer's scores are a learned scorer's outputs over a narrow range — precisely the distribution that clusters into few fp16 buckets. On such rows the descent costs up to 4 extra *global* full-row passes and both designs go to 2–5x, while **#55122's cost is flat and data-independent at 1.14–1.30x**. For a kernel on the decode critical path, a flat 1.2x can beat a bimodal {1.05x, 3x}. **Cheapest possible check, do it first:** add one `atomicAdd` counter on the `bin_pop &gt; MAX_BUFFERED_ITEMS` branch of #55314, run a real Flash-Next agent turn, and read the trigger rate. If it fires on more than ~10% of rows, this whole design is not worth building and #55122 should win.

**5.2 Measure the amortisation before spending days.** This is one op inside the QSA indexer. If the top-k is ~3% of a decode step, then 1.05x versus 1.30x is 0.75% end-to-end — below the restart-to-restart spread this box already shows on spec-decode timings. Get the kernel's share of a decode step from one nsys capture before committing.

**5.3 Two cheaper interventions that capture most of the win with none of the risk.** Both are changes *inside #55122*, which is already 81/81 exact and reproducible:

- **Raise `RADIX_THRESHOLD` from 16384 to 20480.** `det_select_row` caches at `6272 + 4n ≤ 97120` ⇒ n ≤ ~22.7k. 20480 fits with margin, keeps more of the regressed band on the cached single-CTA path, and is a one-constant change. Measure the 24576-wide shapes.
- **Give `det_select_row` a coarse first pass.** Replace pass 0's `key &gt;&gt; 24` bin with the 11-bit `decode_bin` bucket (valid by the same monotonicity argument), keep caching the fp32 keys, and compact the surviving bucket's indices into a smem list so passes 1–4 iterate over the survivors instead of all n. That is ~40 lines inside one function, no new kernel architecture, no new bitmaps, and it attacks the actual cause of the 1.14–1.30x identified in §1. If that recovers most of the gap, the design above is redundant.

**5.4 Diff size and upstream odds.** The full design touches five kernels (`histogram_2048_topk`, `histogram_256_topk`, `radix_topk`, `FilteredTopKUnifiedKernel`, `topk_histogram_4096.cuh`), changes two shared-memory layouts, changes `kSmemMedium`, and rests on an fp16/fp32 monotonicity invariant that is invisible at the call site. Two PRs are already open against this file. A third that supersedes both is a hard review and will need rebasing onto whichever lands.

---

# 6. Verdict

**A good merged design does exist** — it is §2, it is genuinely ~1.05x rather than 1.14–1.30x, and it is *strictly more exact than #55314* (no arrival-order clip at any tie multiplicity, because the tie set never needs to be materialised). It also removes #55122's 1.54x cell outright by putting `RADIX_THRESHOLD` back to 32768.

**But I would not build it yet, and if forced to ship today I would ship #55122.** #55314 is 0/81 on both self-consistency and index-canonical order — it fixes the half of the bug that never produced the user-visible symptom, and it cannot be combined with a cheap order fix, because its arrival-order tie truncation *is* a set defect, not just an order defect. #55122 is 81/81 on everything and 16/16 exact. Its measured cost is real but its worst cell is an artifact of a forced threshold change, not of determinism.

Order of work I would actually take: (1) the descent-trigger-rate counter from §5.1 and the nsys share from §5.2 — half a day, and either result can end the project; (2) the two cheap interventions in §5.3 inside #55122; (3) only if the 1.54x cell and the 1.30x cell both survive that, build §2.