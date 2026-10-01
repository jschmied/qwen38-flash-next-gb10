# T13: is TensorFold's EXL3 routed-expert prefill decode-bound? (2026-10-01)

Flash Next: D 2560, expert width 640, 512 routed + 1 shared expert (11 slots a row), codebook mul1, ~3 bpw (K2 6).
`grouped_kernel` (cuda/exl3/experts_grouped.cuh) decodes each weight fragment per 16-row M tile and uses it for one
mma.m16n8k16; prefill calls it in windows of 1,024 rows (~20 rows per routed expert, all rows for the shared one).

Microbenchmark (`grouped_bench.py`, synthetic trellises at the real shapes; decode cost does not depend on the values):
A. stock `routed()` per window of 1,024 / 2,048 / 4,096 rows, and the two `grouped` launches alone (CUDA events).
B. decode once: `dequant` every used expert's three matrices once per window (+ its time alone), then a padded `bmm`
   in fp16 over each expert's rows (option 2's ideal; not bit-exact, timing only).
Hypotheses (before running):
- A: per-row time flat within ±15 % across windows (decode per M tile, so rows per expert do not amortize anything).
- grouped launches are ≥ 70 % of `routed()`.
- B decode-only time for one window < 25 % of A's grouped time (decoding once is cheap); B total (decode + bmm) is
  2–4× faster than A's grouped time at 1,024 rows. If B is < 1.5× faster, decode is not the bottleneck: stop T13.

## Result 1 and round 2 (2026-10-01)

bench2 (K2 6, R 1024 only — R 2048 fails in `group`: shared memory R·11·4 B > 48 KB, the MOE_WINDOW limit, #151):
grouped launches 21.1 ms of `routed()` 26.6 ms (79 %), 20.6 µs per row. Decode-once oracle 59 ms = 0.35× (slower):
fp16 materialisation moves ~5 GB per window against ~0.95 GB of 3-bit trellis. **Option 2 is dead.**
But the kernel is far above both floors (trellis read ~3.5 ms at ~270 GB/s, decode ALU a few ms). New suspect from the
launch code: grid.z = mats·splits·⌈maxm/16⌉ with maxm = the window (1,024) → 64 row tiles per expert where a routed
expert uses ~2: ~97 % of 1.3 M gate/up blocks (657 k down) start, read `members`, and return.
Round 2 (`grouped_bench2.py`): routed experts only, R 1024, `grouped` with members [maxu, 1024] (stock) vs the same
members compacted to [maxu, max routed rows] (same non-empty tiles, fewer empty blocks); shared expert alone separately.
- compacted ≥ 3× faster than stock on both projections; Z partials bit-identical for every member row.
- shared expert alone (1 expert, 1,024 rows, 64 tiles) costs < 15 % of the routed launches.
If compacted is < 1.5× faster, empty blocks are not the cost: go back to the decode-per-tile question.

Round 2 result: routed-only compacted (maxm 1024 → 48) 20.66 → 13.87 ms = 1.49× (below the ≥3× guess, at the 1.5
stop line), bit-identical; shared alone 2.45 ms (11.6 %, in range); routed+shared in one call 1.00× (the shared
expert's 1,024 rows force maxm = window). Fix A (no bit change): shared expert in its own call / a device tile list →
~21 → ~16.3 ms (−23 %).

## Round 3: is the rest per 16-row tile (decode + trellis read per tile)?
Routed only, compacted, R 1024, top-k 4 / 8 / 12 / 16 (mean rows per expert 8 / 16 / 24 / 32, tiles ~1 / 1–2 / 2 / 2–3).
- If time ∝ Σ tiles (not rows): per-tile cost dominates → option 1 (decode once for 4 sub-tiles) roughly halves the
  routed time at ~20 rows. Expect ms per tile flat within ±20 % and ms per row falling as rows per tile rise.
- If time ∝ rows: mma/activation side dominates → option 1 gains little; stop at fix A.

Round 3 result (partial: top-k 12/16 overflow the same 48 KB group limit): top-4 6.12 ms (512 tiles), top-8 11.18 ms
(726 tiles), top-10 13.87 ms; a·tiles + b·rows fit (3.6 µs/tile, 1.05 µs/row) predicted top-10 within 4 %, but ncu
shows the "per row" reading is wrong:

ncu (`grouped_ncu.py`, `data/tfexl3/grouped-details.csv`): gate/up stock 14.8 ms / compacted 10.0 ms, down 7.1 / 4.6;
memory throughput 12–14 %, SM throughput 17–19 %, **167 registers/thread → 3 blocks/SM → 25 % theoretical occupancy**,
~16 cycles per issued instruction, tensor-pipe instructions 2.2 % (stock) / 3.3 % (compacted) of all. The kernel is
latency-bound: few warps, an instruction stream of decode + indexing (~30 instructions per mma).

## Round 4: occupancy
Worktree `~/git/tf-exl3-prefill` (from c464617), own extension cache. `grouped_kernel` with `__launch_bounds__(W*32, B)`
B = 4, 5 (registers capped at ~128 / ~102). Same arithmetic → Z bit-identical expected.
- B=4: grouped launches −15…−35 % if no spills; B=5 better or spills (check -Xptxas -v / ncu local memory).

Round 4 result: register cap null/worse (MINB 4: routed compact 13.94 vs 14.00 ms; MINB 5: 16.11, spills 600–1,800 B
in some instances), hashes identical across builds. Rejected.

## Round 5: tile work list (branch `exl3-prefill`, worktree `~/git/tf-exl3-prefill`)
`group_kernel` also writes `tiles[]` ((place << 8) | member tile, every non-empty tile) and `tcount`; `grouped_kernel`
takes its expert and member tile from the list; grid (tile bound, N blocks, mats·splits), bound ⌈R·slots/16⌉ + maxu.
`routed_bench.py`: `routed()` on the real path (10 routed + shared, R 1024; and R 1/8/64 for decode sizes), stock vs
branch, output hash.
- R 1024: `routed()` 26.6 → 21.5…23 ms (−13…−19 %); grouped part ~21 → ~16.5.
- R 1/8/64: within ±5 % (few tiles either way; one extra small store per used expert in `group`).
- Output hashes identical between stock and branch at every R.

Round 5 result: routed() R 1024 26.42/26.39/26.41 → 18.69/18.67/18.70 ms (−29 %, beat −13…−19: the shared expert's
64 tiles no longer multiply every expert's grid), R 64 −8 %, R 1/8 ±0; hashes identical at every R; TF's EXL3 GPU tests
on the branch 69 passed / 51 skipped (incl. GLM bit-identity and graph replay).

## Round 6: end to end — prefill on the real EXL3 checkpoint
`prefill_ab.py`: one engine per arm, warm 1k, then fixed prompts (stdlib source, deterministic first line per size) of
8,192 and 32,768 tokens, 16 tokens each; prefill_s from TF's stats + a hash of the 16 tokens. Stock vs branch,
alternating, two rounds.
- 8k prefill −10…−18 %; 32k −10…−18 % (per-row cost, same share); token hashes identical across arms.

## Round 7 (after #184): where does routed()'s remaining time go?
`routed_profile.py` on the #184 branch (0.6.1 + tile list), torch.profiler per kernel, R 1024 and 2048 (#151 allows
2048), Flash Next shapes. Guesses before measuring: grouped gate/up + down ~75 % of routed(); group_kernel (one block,
every thread scans all R·11 picks for its 4 experts) 5–10 %; rot_in + epilogues the rest. At 2048 rows per call the
grouped share per row falls 0–10 % (more rows per expert, same per-tile decode).

Round 7 result (#184 branch, routed() per kernel): R 1024 18.97 ms = grouped 13.63 (72 %), group_kernel 2.53 (13 %),
epilogues 2.17 (11 %), rot_in 0.64 (3 %); R 2048 34.69 ms = 16.9 µs/row (−8.5 % per row). routed() ≈ 71 % of an 8k
EXL3 prefill.

## Round 8: lever 1 — MOE_WINDOW 1024 → 2048 (branch `exl3-window-2048` on #184)
prefill_ab.py, #184 vs #184 + window 2048, alternating, two rounds; Flash Next EXL3 GPU test on the branch.
- 8k / 32k prefill −4…−7 %; first 16 tokens identical (rows never depend on the window).

Round 8 result: MOE_WINDOW 2048 on #184: 8k 10.40/10.30 → 9.90/9.92 s, 32k 42.29/41.57 → 40.26/39.86 s (−3.2…−5.7 %),
token hashes identical, qwen4_exp EXL3 test passes.

## Round 9: lever 2 — parallel grouping (branch `exl3-group-parallel` on #184)
count (atomics) → one-block scan over experts (places, tiles) → a warp an expert compacting its picks with ballots.
- group: 2.53 → < 0.3 ms at R 1024 (5.07 → < 0.5 at 2048); routed() 18.97 → 16.4…16.9 ms.
- routed() hashes = #184's at R 1/8/64/1024 (members identical → everything downstream identical).
- EXL3 GPU tests (incl. group smem file, refusal test rewritten) pass.

Round 9 result: grouping 2.53 → 0.083 ms (R 1024), 5.07 → 0.16 (R 2048); routed() 18.71/18.88 → 16.35/16.38 ms
(−13 %), R 64 −3.5 %, R 1/8 noise (R 1 swings 0.080–0.109 ms between processes); hashes identical; 76 EXL3 GPU tests pass.
