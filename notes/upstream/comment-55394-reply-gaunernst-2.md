DRAFT — needs the user's go. Reply on RFC #55394 to gau-nernst (2026-09-05 06:44). Numbers complete (findings 120/121). Part 3's e2e numbers for the tuned table come from `tuchoice`.

Agreed on all three points, and thanks for the direction — numbers below.

**1. Microbenchmarks, on captured selections.** Yes: we dump the indexer's real top-k output (`block_indices`, `visible_blocks`, query positions) from a server forward and replay it through the kernels with random K/V. Same three chunks every time (an 8k prefill chunk of 3,813 rows, a 3,407-row chunk at 7.5k context from a 30k prefill, a 283-row tail chunk at 12k), medians of 5×5 timed launches, outputs within 1e-4 of the split-K kernel:

| chunk | split-K | tile-union kernel | whole path (pack + sort + build + kernel) |
| --- | --- | --- | --- |
| 8k, 3,813 rows | 14.6 ms | 5.4 ms (2.7×) | 6.5 ms (2.24×) |
| 30k chunk, 3,407 rows at 7.5k ctx | 14.2 ms | 7.3 ms (1.9×) | 8.3 ms (1.71×) |
| tail, 283 rows at 12k ctx | 1.19 ms | 0.73 ms (1.6×) | 0.84 ms (1.41×) |

(`tools/qsa_union_test.py` in the repo linked from the RFC; data `notes/data/qsaunion15.txt`, `tuval*.txt`.) One correction to those numbers that I only found while preparing this: the 14.6 ms baseline is the **pre-#54873** split-K kernel — our nightly (`dev401`, 8340fe1bb) was built an hour before your improvement merged, and the venv carried the old kernel until we overlaid the branch onto it. Your kernel runs the same 8k chunk in 9.4 ms. Against it the union is 1.50× (6.3 ms) on the 8k chunk and 1.42× on the 7.5k-context chunk (`tools/qsa_three_way.py`, same dumps, cache spread over 64 pages or not — no difference, so this is not an L2 effect). So the honest kernel-level number for the union over current main is ~1.5×, not 2.7×.

**2. Where the 2.7× goes end to end — torch-profiler traces of one 7.5k-token request in the server, union on vs off, same branch:**

| | union on | off |
| --- | --- | --- |
| QSA attention kernel, 24 calls (2 chunks × 12 layers) | 182 ms (7.6 ms/call) | 265 ms (11.0 ms/call) |
| kernels between top-k and attention | 10.4 ms (sort 3.8, pack 1.4, build 1.2, layout ops 2.0) | 6.3 ms (expansion 4.2) |
| GPU idle in that window | 3.6 ms (0.15 ms per call, 3 µs launch gap) | 0.2 ms |
| all QSA-related kernels, share of GPU time | 7.4 % of 2.55 s | 10.5 % of 2.59 s |
| TTFT of the profiled request | 2.64 s | 2.69 s |

So in situ the union kernel is 1.45× the split-K kernel — consistent with the corrected replay — and the integration costs nothing (0.15 ms idle per call, 3 µs launch gap): −83 ms of kernel, +4 ms of glue, +3 ms idle = −76 ms = the 2.9 % we see. The attention is 10 % of prefill, so ~3 % is this design's ceiling on this box. I would not merge #55430 on that basis either; I'll mark it draft and leave it as the reference for the design, unless the SM120/GB300 numbers someone else collects with the override say otherwise.

**2b. Short-context boundary, where #54873's pruning is strongest** (same branch, union vs off, three reps, pair wall): 1×1,521 tokens 0.62 vs 0.62 s, 1×2,031 0.77 vs 0.78 s, 1×4,106 1.45 vs 1.49 s; batches under the 1,024-row gate run stock in both arms and match. So the union does not lose where every row's selection is still short of the budget, and starts paying from ~4k where the selection saturates.

**3. Retuning the split-K kernel on GB10 — done as a sweep, and it is the better first PR.** `_select_config` swept over BN ∈ {16, 32, 64, 128} × warps ∈ {1, 2, 4, 8} × target splits ∈ {1 … 64} per dispatch region, on the same captured prefill chunks plus synthetic uniform decode/verify batches (one run per cell so far):

| shape (rows × requests) | base programs | stock (BN, splits, warps) | best on GB10 | gain |
| --- | --- | --- | --- | --- |
| 1 × 1 | 2 | 32, 64, 4 | 32, 16, 1 | 1.02× |
| 4 × 4 | 8 | 32, 64, 4 | 32, 8, 1 | **1.61×** |
| 16 × 4 | 32 | 32, 16, 1 | 64, 1, 2 | **1.34×** |
| 32 × 8 | 64 | 32, 8, 1 | 64, 1, 4 | 1.12× |
| 64 × 16 | 128 | 32, 4, 1 | 64, 1, 2 | 1.11× |
| 128 × 32 | 256 | 32, 8, 1 | 64, 1, 2 | **1.21×** |
| 512 × 128 | 1024 | 64, 1, 2 | same | 1.01× |
| prefill chunks | > 2048 | 32, 1, 1 | 32, 1, 8 | 1.05× |
| prefill tail, 283 rows | 566 | 64, 1, 2 | 16, 1, 4 | **1.25×** |

The GB300 table over-splits on 48 SMs: 64 splits at ≤ 24 base programs is 512 tiny programs plus the merge; from 32 base programs up, no split with BN = 64 wins. At the server (one start each, same branch, `--max-num-seqs 16`): the GB10 table alone is −1.5 % TTFT at 7.5k / −1.3 % at 29k and +4 % no-spec decode at 4 streams (67.7 vs 65.1 tok/s); the tile-union on top of it is a further −1.9 % / −1.4 % TTFT (2.57 s / 10.10 s vs 2.66 / 10.38 stock) — the two stack, since one changes the kernel's tile and the other what it gathers. I'll open the table as a short PR — a CC 12.x entry in `_select_config` (keyed the same way you'd key GB300) plus the sweep script under `benchmarks/kernels/` — after repeating the cells three times, filling the two shapes I skipped (256 and 2048 rows), and running the server A/B (decode c=1/4/16 and TTFT), since the decode cells here are small enough to be L2-resident and the merge behaves differently under CUDA graphs. A few days.
