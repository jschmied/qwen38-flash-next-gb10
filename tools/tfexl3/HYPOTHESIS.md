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

## Round 10: lever 2 rebased on #184 head (`0b851ce`) — end to end
prefill_ab 8k/32k #184 vs #184 + parallel grouping (`f71d25f`), alternating, two rounds; EXL3 GPU tests (incl. the
review's reuse test); small_bench R 1–16 both, three rounds.
- prefill −7…−10 % (routed −13 %, ~71 % of prefill), token hashes identical.
- small windows: equal or faster than #184 (grouping cheaper at every R).

## Round 11: lever 3 — two member tiles a program (branch `exl3-subtiles` on #193)
Windows ≥ 512 rows: a program covers two consecutive 16-row member tiles of one expert (MS 2) with 4 n tiles (NT 8 ×
MS 2 would need 64 KB of static reduction memory); each decoded fragment feeds 4 mma instead of 2; K order, splits and
warp reduction unchanged. Decode windows keep MS 1 / NT 8.
- grouped launches −20…−35 % at R 1024 (13.8 → 9–11 ms); routed() −15…−25 %; prefill −10…−18 % vs #193.
- routed() hashes identical at R 1/8/64/1024 (1024 = 83704058100e6b60); MS2-vs-MS1 test bit-equal.
- < 10 % on the grouped launches → stop (activation loads per mma double with NT 4; may cancel the decode saving).

Round 11 result — STOPPED by the rule (negative): MS 2 / NT 4 bit-identical (78 tests) but grouped +16 % (R 1024) /
+64 % (R 2048); ncu: instructions −26 %, tensor +11 % (empty halves of partial pairs), L1 bytes +24 %, cycles per
issued instruction 14.3 → 23.0, occupancy 25 % both. Latency-bound: decode reuse only pays with an ILP redesign.

## Round 12: lever 4 — epilogues and rot_in in larger blocks (branch on #193)
After #191: gateup_epilogue 1.10, down_epilogue 1.07, rot_in 0.64 ms of routed() 16.4 at R 1024 (17 %); all launched as
one warp a (row, 128-column block): 56 k / 225 k / 450 k blocks. Fold 8 rows (or 8 column blocks) into one 256-thread
block, same per-element arithmetic and order.
- the three kernels 2.8 → 1.0…1.6 ms; routed() −7…−11 %; hashes identical.

## Round 12 (revised): lever 4b — the down epilogue writes the prompt's bf16 rows (branch `exl3-epilogue-blocks` on #193)
8k profile on #193 (`p8k`): grouped 54.5 %, dense prompt GEMMs 13.1 %, epilogues 9.2 % (at DRAM bandwidth already, so
bigger blocks were dropped), `bfloat16_copy_kernel` 3.0 % (forward's per-window fp32 → bf16 copy of routed()'s rows),
rot_in 2.7 %, HC glue 6.2 %. Change: down_epilogue stores bf16 directly into buf.y (same fp32 value, RNE), no copy.
- prefill 8k/32k −2.5…−4 %; tokens identical; new test bf16-out == fp32.to(bf16).

Round 12 result: lever 4b 8k 9.62/9.61 → 9.19/9.09 s, 32k 38.77/38.77 → 37.20/36.72 s (−4.1…−5.5 %), tokens identical.
Window 2048 stacked on 4b: 8k 8.92/9.21 → 8.76/8.62, 32k 36.08/37.27 → 36.01/34.86 — sign holds, but round a 32k
(−0.2 %) is below the base arm's own 3 % drift: inconclusive, parked.

## Round 13: lever 5 — 64-row tiles for the prompt's fp16 matmuls (`_f16_mm`, 6.0 % of 8k)
`F16.prefill` used the decode tile (BM 16); BM 64 only groups rows. Microbench (`f16_bits.py`, 5 shapes): bit-identical
at every shape; −34…−50 % on the large mixes, ±0 on the small. E2E (prefill_ab, vs `exl3-bf16-rows`, two rounds):
- 8k/32k prefill −2…−4 %; tokens identical.

## Round 14 (D): settle the two parked levers — 4 arms × 3 alternating rounds, 8k + 32k
Arms on top of #195 (`exl3-bf16-rows`): base, + fp16 prompt tile (`exl3-f16-prompt-tile`), + window 2048
(`exl3-stack-win`), + both (`exl3-f16-win`). Rule: a lever counts only if its sign holds in all 3 rounds AND its median
gain beats the base arm's own spread (max − min over rounds).
- f16 tile: 8k −2…−4 %, 32k −1…−3 %; window: −1…−4 %; both ≈ additive; tokens identical in every arm.

## Round 15 (2026-10-02): TF PR #212 (grearjake-star, EXL3 prompt experts) on our GB10

Arms: 0.6.1 (`17c73e1`), #212 (`9f062b2`, on 0.6.1), our stack head `exl3-stack-win` (#184→#207, on pr-141-0.6.1).
Flash Next EXL3 3.05 bpw, `prefill_ab.py` (8k + 32k code prompts, 16-token hash), three alternating rounds; then
decode (`tffp4x/accept.py`, 8 prompts x 256 tokens, drafts on) for 0.6.1 vs #212, two rounds.
- #212 vs 0.6.1: prefill −45…−60 % at 8k and 32k (their 6K/24K: −51/−52 % on 3.05); our stack −25…−28 %.
- Hashes identical across all three arms (all claim same bits).
- Decode: #212 within ±2 % of 0.6.1 (decode windows keep the grouping kernel).
- Their tests + our EXL3 tests pass on GB10.
- **Result (§5be):** #212 −53.5…−54.6 % (in range), stack −22.6…−25.0 % (slightly below range), hashes identical,
  decode level (71.8–72.0 vs 71.9 tok/s, identical rounds).

## Round 16 (2026-10-02): does our grouping stack (#184 + #191 + #193) still add on top of #212, in decode?

#212 keeps decode/verify windows (≤ 64 rows) on the grouping kernel. Arms: #212 (`9f062b2`) vs #212 + our 4 commits
(`53da6ed`, branch `pr212-grp`). Decode `tffp4x/accept.py` (8 prompts x 256 tokens, drafts on), three alternating rounds.
- Prediction: 0…−3 % ms per round (50–70 pairs over ~60 experts: few empty tiles, cheap grouping); tokens identical.
- Below the base arm's spread or a sign flip: no decode value → close #184/#191/#193 with #195/#207.
- **Result (§5be):** no decode effect (−0.1 / −0.1 / +0.2 %, sign flips, inside the 0.65 % spread) → close all five.
