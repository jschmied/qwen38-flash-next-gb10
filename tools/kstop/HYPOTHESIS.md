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
