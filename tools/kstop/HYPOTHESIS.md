# Deeper drafts + confidence stop + graphs per draft count (2026-09-29, before any run)

User: "yes and also try additional capture graphs, i expect 3 to be hit often" (after: K=7 = 1 + 7 = 8 rows, stop
thresholds that depend on whether the next draft opens a new graph size).

Mechanism check (read, not run): under model runner V2 a verify at fixed K always replays a uniform-decode FULL graph of
exactly 1+K rows per request (`cudagraph_utils._init_candidates`), so odd entries in `cudagraph_capture_sizes` change
nothing at fixed K. Variable row counts only appear with a stop rule. MRV2's *dynamic speculative decoding*
(`num_speculative_tokens_per_batch_size`) already captures FULL graphs for several decode query lengths in one server.
Those are the "graphs per draft count" a confidence stop would select from.

Three runs, one chain (clone venv `rssm`, prod config: K=5 base, probabilistic drafts, 32k NVFP4 draft slice,
RecoverSSM, bf16 state, pmu 64, HC fusion, FULL_AND_PIECEWISE, KV 4 GiB):

1. **dl7** — draft log at K=7 (PIECEWISE: the Python logging hook does not run inside FULL-graph replays), 8 prompts ×
   700 tokens, 1 start. Feeds the offline replay of depth 5–7 × τ.
2. **kcost** — K = 2…7, each with its own exact graph, 2 starts each, nvprobe. Per verify cycle (ms/tok × accepted):
   predicted code K2 50–54, K3 54–58, K4 58–62, K5 63–65 (measured 63.8–64.5), K6 67–71, K7 71–76 ms; prose the same
   cycle cost within ±2 %. Throughput without a stop: code K5 best or K6 within ±2 %, K7 −1…−4 % vs K5; prose best at
   K3 (as §5l), K7 −8…−12 % vs K5. The fitted cycle(K) replaces the replay's assumed 43 + K·(1.3 + 3.3) ms.
3. **dynsd** — one server with the schedule bs1→K7, bs2→K6, bs3→K5, bs4→K4, bs5→K3, bs6–16→K2, so FULL graphs for
   3…8 rows are captured side by side. Predicted: capture succeeds for all six lengths (risk: the RecoverSSM builders or
   the drafter may assume one K; if so, that is the first blocker to record); startup +10–40 s; c=1 decode equals the
   static K=7 arm within ±2 % (the scheduler uses K=7 at bs 1).

Replay after (offline, no GPU): depth 5/6/7 × τ, three designs:
(a) stop anywhere, exact graphs for every draft count (what dynsd provides);
(b) round the stop up to 3/5/7 drafts (only even row counts, 4/6/8);
(c) two thresholds, τ_new when the next draft opens a new graph size and τ_padded when it fills a paid row.
(b) and (c) are bracketed over padded-row cost 0 / 1.1 / 3.3 ms because that residual is unmeasured. The (a) vs K5
baseline predicted: code +4…+7 %, prose +8…+14 % (deeper drafts help code only with the stop).

## Async-scheduling cost (`noasync`, 2026-09-29, before the run)

Why: an exactly sized verify needs the scheduler to know each request's draft count, which under async scheduling it
cannot (the AsyncScheduler sets next-step spec placeholders at schedule time, before drafting). The exact-size design
therefore runs with `--no-async-scheduling`; the async-compatible alternative (8-row verify, stopped rows padded) only
pays if a padded row costs <= ~0.5 ms (replay: code +7.1 / prose +6.2 % at 0.5 ms, prose negative at 1.5 ms).
Arms: prod config at K=5 (per-K capture list), async (default: MTP is an EAGLE type, so vLLM enables it) vs
`--no-async-scheduling`; 2 starts each, nvprobe. Witness: `'async_scheduling': False` in the non-default-args line.
H: no-async costs +2…+6 % ms/tok at c=1 (step n+1's scheduling and input prep no longer overlap step n); TTFT ±1 %;
greedy hashes equal. Decision: if the cost is <= ~2 %, the exact-size design (+8.4…10.3 % before it) wins.

## Overhead diagnosis (`kdiag`, 2026-09-29, before the run)

Smoke `kstop3` (τ 0.75): accepted tokens per cycle match the replay (code 4.78 vs 4.56, prose 2.44 vs 2.46) but the
cycle takes longer (code 68.1 vs 63.8 ms, prose 57.6 vs 55.2): ~2.4–4.3 ms of machinery overhead. Arms, 1 start each,
all `--no-async-scheduling`: static K7; kstop τ=0 (never stops, d=7 always); kstop τ=1.1 (always d=1); static K1.
H1 (machinery at full drafting: IF-node conditional + confidence softmax + d copy/sync + per-K verify graphs):
kstop τ0 − static K7 = +0.5…+2 ms per cycle. H2 (host prep of skipped steps: per-step attention metadata and graph
launches still run on the CPU while the GPU has nothing to do): kstop τ1.1 − static K1 = +1…+4 ms per cycle.

## A/B `kstopab` (2026-09-29, before the run)

kstop in runner mode (τ 0.75, depth 7, dynamic-SD graphs, async scheduling ON as in prod) vs prod K=5, 2 starts each,
alternating. From the single starts (`krun`): H: code c=1 −5…−8 % ms/tok, sampled code −5…−8 %, prose −3…+1 %,
TTFT ±1 %, c=4 ±5 %. Greedy text differs from K=5 (RecoverSSM commit grouping, §5b), identical between kstop starts.

## `kstopab2` (2026-09-29, before the run): prod-shaped schedule

Schedule bs1–10 → K7, 11 → 6, 12 → 5, 13 → 4, 14 → 3, 15 → 2, 16 → 1 (all draft counts captured, max-num-seqs stays 16).
vs prod K5, 2 starts each. H: c=1 as `kstopab` (code −6…−7 %, prose −1…−3 %); c=4 now within ±3 % of prod (K7 + stop at
batch 4 instead of K4); TTFT ±1 %.

## TODO 10a: block verification (`blockver`, 2026-09-29, before the run)

`rejection_sample_method: "block"` (myllmbox v4) on top of the confidence-stop config, vs without; 2 starts each, clone
venv. Block verification only changes the sampled acceptance rule (greedy verify is exact-match either way). H: code
sampled −2…−6 % ms/tok (more accepted per cycle), greedy code / prose unchanged within ±1 % and greedy hashes identical;
TTFT ±1 %. If sampled gains < 2 %, not worth a prod change.

## `ksprod` (2026-09-29, before the run): the stop in prod's memory regime

prodval5 (prod venv, fusions + stop, default KV) gave code c=1 14.16 vs prodval4's 14.26 ms/tok (−0.7 %, not −7 %) and
MiaAI's quicksort greedy 52.2 vs 66.4 tok/s. All A/Bs ran at KV 4 GiB; at prod's default KV the mapped PLE table pages
(memory note "measure with reduced KV": ~30 major faults per step), and K7 verifies up to 8 rows (more PLE rows per
step). Arms on the prod venv with prod's exact env (fusions on), default KV: stop (drop-in 65 env) vs K5; 2 starts each;
nvprobe + MiaAI prompts. H1 (paging): the stop's code gain at default KV is ≤ 2 % and MiaAI quicksort is slower with the
stop in both starts. H0: prodval5 was one noisy start; the stop gives −5…−8 % here too.

## TODO 10e: vllm#58449 fused draft metadata (`f58449`, 2026-09-29, before the run)

Prod K5 on the clone venv (KV 4 GiB, no stop), #58449 on vs off (armrun source_toggle), 2 starts each. Witness:
`FN58449 fused QSA draft metadata update ran` in the on arm; `Fused multi-step draft decode is not supported` only in the
off arm. H: greedy hashes identical (the same metadata computed in place); code c=1 −1…−4 % ms/tok (no host metadata
rebuild between the 4 decode draft steps); prose similar; TTFT ±1 %; c=4 ±3 %.

## `kssplit` (2026-09-29, before the run): text vs KV size

Same as `ksprod` (prod venv, fusions on) but `--kv-cache-memory-bytes 4294967296`; stop vs K5, 2 starts. With `ksprod`
this is a 2×2. H-KV: the stop regains −5…−8 % code c=1 and MiaAI quicksort ≥ K5 at 4 GiB → the loss is the default-KV
regime (PLE paging). H-text: the stop stays within ±2 % at 4 GiB too → the loss is GDNNQ's text.

## TODO 10c: Marlin MoE at decode (`marlin`, 2026-09-29, before the run)

Clone venv, K5, KV 4 GiB, fusions off (GDNNQ and FNMOEFUSE unset: the text equals the K5 baseline and the Triton
prefill MoE only hooks the FlashInfer path), `--moe-backend marlin` + `VLLM_MARLIN_USE_ATOMIC_ADD=1` vs FlashInfer
CUTLASS; 2 starts. H: decode c=1 within ±3 % (the experts' bytes are the same 4.5 bpw; Marlin dequantizes to bf16);
TTFT +10…+30 % with Marlin (bf16 math at prefill); greedy hashes may differ (different MoE kernel numerics).

## Agenda 2b: GDNNQ warm replay (`gqreplay`, 2026-09-29, before the run)

Prod venv, prod env with FNMOEFUSE, KV 4 GiB, FN_GDNNQ on vs off, 2 starts; probe `tools/gdnnq/replay1.py` (the fixed
8k replay prompt with 1 output token: cold, then 3 warm; and with 96 tokens as nvprobe). H0 (text): warm 1-token TTFT
equal within ±3 % (the 1.81 vs 1.52 s gap is GDNNQ's different 96 decoded tokens); cold 1-token TTFT −1…−3 % with
GDNNQ (§5af). H1 (regression): warm 1-token TTFT slower with GDNNQ by > 5 % in both starts → propose removing GDNNQ.
