DRAFT — needs the user's go. Reply on RFC #55394 to gau-nernst (2026-09-05 06:44). Fill the [OFF] numbers from `tuprof` (PROF0 arm) before posting.

Agreed on all three points, and thanks for the direction — numbers below.

**1. Microbenchmarks, on captured selections.** Yes: we dump the indexer's real top-k output (`block_indices`, `visible_blocks`, query positions) from a server forward and replay it through the kernels with random K/V. Same three chunks every time (an 8k prefill chunk of 3,813 rows, a 3,407-row chunk at 7.5k context from a 30k prefill, a 283-row tail chunk at 12k), medians of 5×5 timed launches, outputs within 1e-4 of the split-K kernel:

| chunk | split-K | tile-union kernel | whole path (pack + sort + build + kernel) |
| --- | --- | --- | --- |
| 8k, 3,813 rows | 14.6 ms | 5.4 ms (2.7×) | 6.5 ms (2.24×) |
| 30k chunk, 3,407 rows at 7.5k ctx | 14.2 ms | 7.3 ms (1.9×) | 8.3 ms (1.71×) |
| tail, 283 rows at 12k ctx | 1.19 ms | 0.73 ms (1.6×) | 0.84 ms (1.41×) |

(`tools/qsa_union_test.py` in the repo linked from the RFC; data `notes/data/qsaunion15.txt`, `tuval*.txt`.) One caveat we found on the way, which is the answer to your "depends on the input data": these replays put the K/V of 18 pages in the cache, i.e. beyond the 24 MiB L2. With a 3-page cache the split-K kernel runs the same 8k chunk in 9.4 ms, not 14.6, and the union's edge shrinks accordingly — the union pays off when the gathers miss L2.

**2. Where the 2.7× goes end to end — torch-profiler traces of one 7.5k-token request in the server, union on vs off, same branch:**

| | union on | off |
| --- | --- | --- |
| QSA attention kernel, 24 calls (2 chunks × 12 layers) | 182 ms (7.6 ms/call) | [OFF] ms ([OFF]/call) |
| union pack + build kernels | 2.5 ms | — |
| all QSA-related kernels, share of GPU time | 7.4 % of 2.55 s | [OFF] % of [OFF] s |
| TTFT of the profiled request | 2.64 s | [OFF] s |

So in situ the union kernel is ~[OFF]× the split-K kernel, not 2.7×: the server's K/V for an 8k prompt (~25 MiB) straddles the L2, the split-K kernel's gathers are cheaper there than in the DRAM-bound replay, and the attention is [OFF] % of prefill to begin with. That is the whole story of the ~3 %: the kernel gain is real, the kernel's share is small, and the replay overstated the gap. I would not merge #55430 on that basis either; I'll mark it draft and leave it as the reference for the design, unless the SM120/GB300 numbers someone else collects with the override say otherwise.

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

The GB300 table over-splits on 48 SMs: 64 splits at ≤ 24 base programs is 512 tiny programs plus the merge; from 32 base programs up, no split with BN = 64 wins. I'll open that as a short PR — a CC 12.x entry in `_select_config` (keyed the same way you'd key GB300) plus the sweep script under `benchmarks/kernels/` — after repeating the cells three times, filling the two shapes I skipped (256 and 2048 rows), and running the server A/B (decode c=1/4/16 and TTFT), since the decode cells here are small enough to be L2-resident and the merge behaves differently under CUDA graphs. A few days.
