# Evidence audit — what is proved, what rests on three runs (2026-09-02)

Standard applied: the repo's own (`evidence-standard.md`: n≥2 for direction, n≥3 for magnitude,
effect must exceed *that config's* spread, never carry an error bar across configurations) plus
the two spreads the repo itself established: **no-spec 42.8–47.7 ms/tok over 12 arms (1.10×)**,
**MTP 1.83–2.07× per start**, and — decisive for everything MTP — **finding 41: the
healthy/broken state is drawn per request, 42 healthy / 54 broken over 96 turns (44 % healthy),
and 3 of 15 starts came up healthy on all 8 turns.**

Consequence applied throughout: **any MTP ms/tok, acceptance, accept-length or depth number taken
before 2026-09-02 with fewer than ~3 starts × 8 turns is one draw of a two-state variable, not a
measurement of the configuration.** That covers roughly 40 published numbers in this repo.

Classes: **EXISTENCE PROOF** (one observation suffices: "X still happens with Y", "the fix does
not eliminate it"); **ADEQUATELY SAMPLED** (≥3 independent starts and effect ≫ spread, or
bit-level with ≥3 repeats in ≥2 shapes); **UNDER-SAMPLED** (an absence claim or a number on ≤3
starts / one start / one request, where a 4th run could plausibly reverse it);
**WITHDRAWN/CORRECTED** (the note already retracts it). Method: every `.md` under `notes/`,
`README.md`, `REPRODUCE.md`, `patches/**/README.md`; `notes/data/*.txt` skimmed for n. Produced
with Claude Code from the files; the classification is mechanical against the standard above.

## 1. `determinism-investigation.md` — findings 1–42

| # | claim (≤15 w) | evidence actually cited | class | reason / what would settle it |
|---|---|---|---|---|
| 1 | A single forward pass is deterministic | 3 identical requests, 1 start; later 6 arms / 2 starts | **UNDER-SAMPLED** (as worded) | True only for the **one** 55-token unchunked shape. Finding 27 shows M=52 and M=3 *single forward passes* diverge. The title over-generalises from one shape |
| 2 | Divergence at 0 % hit rate ⇒ align machinery is the cause | logged 0.0 % hits, 2 arms × 3 reqs | **WITHDRAWN/CORRECTED** (by 27, not in place) | Finding 27: machinery's role was *chunking the prefill*; "exonerated as a cause". Finding 2 text unamended |
| 3 | Generation diverges without prefix cache | 1040 tok, 3 of 3, 1 start | **EXISTENCE PROOF** | |
| 4/4b | Divergence enters at layer 1 | 2 groups × 3 passes | **WITHDRAWN/CORRECTED** in place | superseded by 25/26 |
| 5 | Spec, cudagraphs, QSA top-k, PLE-FP8, bounds masks excluded | 1 arm each × 3 reqs | **EXISTENCE PROOF** | Exception: "PLE FP8: 0 dead rows in 2100 sampled" is 0.01 % of the table → **UNDER-SAMPLED** as an absolute |
| 6 | MTP not reproducible across restarts, 1.83× | 66 pooled arms, 5 repeated configs | **ADEQUATELY SAMPLED** | |
| 6b | k=2 slower than no-spec: 4 arms ≥47.8 vs 12 ≤47.7 | 4 MTP2 starts × 8 turns, timing only | **UNDER-SAMPLED** | ~20 % of MTP starts are 8/8 healthy; P(4 k=2 starts all miss the fast regime) ≈ 0.4 by chance. **No per-turn acceptance exists for any k=2 arm.** Settle: 3 more k=2 starts with per-turn acceptance |
| 7 | `VLLM_BATCH_INVARIANT` cannot run on this arch | code + one RuntimeError | **EXISTENCE PROOF** | |
| 8 | Acceptance is the channel; r = −0.964 | 5 starts, MTP n=5 | **ADEQUATELY SAMPLED** (direction) | the r value from n=5 is fragile; the direction is not |
| 9a | Per-turn bimodal flip with repeated recovery | 3 starts × 40 turns | **ADEQUATELY SAMPLED** | |
| 9b | Block-boundary crossings trigger collapse | DEG_a only | **WITHDRAWN/CORRECTED** in place | |
| 9c | Lock-in after turn 28 is one-way | 1 of 3 arms | **UNDER-SAMPLED**, self-labelled | Settle: 3 more 40-turn arms |
| 9d | The flip's bias is per-start (12–55 %) | 3 × 40 turns | **ADEQUATELY SAMPLED** for "rate varies between starts" | finding 41 reframes this; the two readings are not reconciled anywhere |
| 9e | **DEG_nospec control is FLAT** | **1 start**, 39 turns, CV 0.05 | **UNDER-SAMPLED** | the load-bearing control for "the base decode path is steady". Settle: 2 more no-spec 40-turn starts |
| 10 | Forced sync after align postprocess does not restore determinism | 1 arm, 3 of 3 | **EXISTENCE PROOF** | |
| 11 | The `'#'`/`'The'` split was our own sort hook | oracle T5/T6 | **WITHDRAWN/CORRECTED** in place | |
| 12 | Async scheduling is not the mechanism (divergence) | NOASYNC, 3 of 3 | **EXISTENCE PROOF** | |
| 13 | Prefill divergence is not a race; closes the race family | LAUNCHBLOCK, 3 of 3 | **EXISTENCE PROOF** for the single test | "closes the family" is an inference; the appended per-start-bias reasoning is dead after 41/42 |
| 14, 15 | 4-token generation deterministic / hook hashes stale rows | — | **WITHDRAWN** in place (by 20) | |
| 16, 17 | `mamba_cache_mode` all/align both diverge | 3+6 arms × 3 reqs | **EXISTENCE PROOF** | |
| 16b | "Scatter, not offset" | mismatched configs | **UNDER-SAMPLED**, self-caveated | |
| 18a | `persistent_topk` clean at 11 sizes ≥512 | 1 sweep, silence-is-pass | **UNDER-SAMPLED**, self-caveated | |
| 18b | Kernel writes −1 into every slot ≥ visible | exhaustive census, 3 sizes | **ADEQUATELY SAMPLED** | |
| 19 | Runtime is not the source | 5 runs per op | **UNDER-SAMPLED**, self-labelled "a bound" | never ran the MoE kernel |
| 20, 21 | Decode diverges from layer 1 / first decode output differs | 1 arm each | **EXISTENCE PROOF** | |
| 22 | `GDNCUDA_a` was a triton replicate | log audit | **EXISTENCE PROOF** (method) | |
| 23 | GDN cuda kernel moves, doesn't remove, divergence | 2 arms × 3 reqs | **EXISTENCE PROOF** | |
| 24 | Layer-0 GDN state deterministic through decode | **1 arm** | **UNDER-SAMPLED** (absence half) | Settle: STATEHASH on 2 more starts |
| 25 | **The PLE is exonerated** | **1 arm**, 3 reqs | **UNDER-SAMPLED** | exoneration is an absence claim from one start |
| 26+27 | `mlp.experts` is the first and only differing module | 2 arms × 3 reqs, 106 modules, M=1/3/52/55 | **ADEQUATELY SAMPLED** | the strongest result in the repo |
| 28 | marlin/humming diverge; cutlass crashes; triton rejected | 1 arm each | **EXISTENCE PROOF** | |
| 30a | `emulation` is fully deterministic | **1 arm, 1 shape** | **UNDER-SAMPLED** | the fix got three shapes; emulation got one. Settle: cache-on and MTP shapes |
| 30b | **Emulation costs +17 % (51.38 vs 43.92)** | **n=1 each** | **UNDER-SAMPLED** | 43.92 is one draw from 42.8–47.7; against the band top it is +7.7 %. Quoted in the upstream drafts |
| 31–33 | Non-fused finalize cannot start (ids 50/50/48) | 3 starts | **EXISTENCE PROOF** | |
| 34 | Non-fused runner has 20 GEMM2 tactics vs 40 | direct probe | **ADEQUATELY SAMPLED** | |
| 37 | Complete fix validated in 3 shapes | 3 arms × 3 reqs × 4 tokens | **ADEQUATELY SAMPLED** | |
| 38 | **The fix costs +3.6 % (45.50 vs 43.92)** | **n=1 vs n=1** | **UNDER-SAMPLED** | 3.6 % against an 11 %-wide band; in the issue draft, report draft and flashinfer comment. Settle: 3 arms each side, interleaved |
| 39 | Cache-key defect fixed upstream from v0.6.18rc2 | file content per ref | **EXISTENCE PROOF** | |
| 40 | Fix does not stabilise MTP; 2.07× | 3 starts, start-level totals | **EXISTENCE PROOF** for "does not eliminate"; the 2.07-vs-1.83 comparison **UNDER-SAMPLED** | |
| 41a | The flip is per REQUEST; two clean states | 12 starts, 96 turns | **ADEQUATELY SAMPLED** | |
| 41b | Async/graphs/ring/cache "ruled out" | 24 turns per group | **EXISTENCE PROOF** for "none eliminates"; **UNDER-SAMPLED** for "none affects the rate" | healthy 8/24, 14/24, 6/24, 14/24 ≈ 1.7 σ apart |
| 41c | Something at start decides whether the broken draw is possible | 3 of 15 starts 8/8 | **ADEQUATELY SAMPLED** vs iid (0.44⁸ = 0.14 %) | turns are not iid (runs/lock-in) — autocorrelation never accounted for |
| 42 | The draw does not follow the prompt | 1 start, 3 identical passes | **EXISTENCE PROOF** | corollary holds within a start only |

## 2. MTP timing / acceptance / depth claims outside the finding list

All measured **before finding 41**; each is one or two draws of the per-request state.

| file | claim | evidence | class | settle it with |
|---|---|---|---|---|
| `speculation-on-flash-next.md` | k=2 +44 % c=1 decode; c=16 +2.6 %; TTFT −30 % "the reason to keep it on" | 1 start/arm | **UNDER-SAMPLED** | 3 starts each arm / 3 pairs at c=16 |
| `depth-curve.md` | **MTP +52–54 % decode, flat 8k→60k**; prefill flat; MTP costs nothing at prefill; "MTP stays on at every depth" | **1 start**, 3 reqs/depth — all depths share one draw | **UNDER-SAMPLED** | the prefill band (19 %) is narrower than one config's own prefill spread (45 %). 3 starts × 5 depths |
| `mtp-vs-prefix-cache.md` | MTP cache cost is a fixed 1600 tokens | counters, 3 starts | **ADEQUATELY SAMPLED** | |
| same | MTP off ~10 % faster on the agent loop; k=1 strictly dominated; **break-even 68 output tokens**; n=6 slower than no-spec | n=1–2 per cell | **UNDER-SAMPLED** | 3 starts per cell |
| `mtp-depth-anomaly.md` | k=2 never reaches the ~32 floor (4 starts) | timing only | **UNDER-SAMPLED** | as 6b |
| same | **n=6 genuinely 2.3× slower; capacity 12 pathological** | 2 arms | **WITHDRAWN** by `failure-modes.md` — still asserted in place | |
| `which-drafter-for-agent-work.md` | ngram n=4 −23 % vs no-spec | ngram n=2, no-spec n=3 | direction **ADEQUATE**, magnitude **UNDER-SAMPLED** | 33.3 sits at the top of ngram's own 28.5–33.8 |
| same | k=2 worst config (47.75); k=2 anomaly prose-specific (36.85 edit-shaped) | n=1 | **UNDER-SAMPLED** | |
| `single-stream-limit.md` | **MTP +67 % at c=1; optimum k shifts with concurrency**; acceptance 77/58 % | n=1/cell | **UNDER-SAMPLED** | anchor of the substitutes argument in `fp8-mixed-checkpoint.md` |
| `fp8-mixed-checkpoint.md` | +71 % with MTP; lm_head × MTP complements; "k=2 remains optimal" | n=1/cell | **UNDER-SAMPLED** (MTP columns) | the no-MTP +39 % column is adequate (§3) |
| `where-the-gpu-time-goes.md` | FP8 head −3.7 pp acceptance | one aggregate | **UNDER-SAMPLED** | +19.1 % under MTP (n=6 vs n=4) is **ADEQUATE** |
| `fused-draft-decode-for-qsa.md` | fused metadata buys nothing at k=3; k=3 2.471 tok/iter | n=1 | **UNDER-SAMPLED** | |
| `prefill-divergence.md` | fixing divergence moves median n=5 58.4 → ~32 | prediction | **WITHDRAWN by finding 40, not corrected** | |
| `mtp-instability-upstream-survey.md` | rank-1 = stale GDN on resume | citations | predates NOCACHE (41) and does not say so | |

## 3. Quantization / checkpoint / config claims

| file | claim | evidence | class |
|---|---|---|---|
| `fp8-mixed-checkpoint.md` | dense FP8 **+39 %** at c=1 (no-spec) | n=1, but 39 % ≫ 6.9 % floor and predicted +32 % by the byte model | **ADEQUATELY SAMPLED** |
| same | FP8-mixed quality: no regression (NLL 0.7748→0.7610) | 276 tokens | **UNDER-SAMPLED**, correctly hedged |
| `quantizing-lm-head.md` | FP8 `lm_head` **+11 %**, no quality cost | 3 cells + 646-tok NLL; confirmed +10.1 % independently | **ADEQUATELY SAMPLED** |
| `quantizing-shared-expert.md` | shared_expert FP8: +1.9 / −2.8 %, NLL +2.06 % worse | n=1 per cell; NLL chunk count not stated | **UNDER-SAMPLED** |
| `block-size-is-not-a-kernel-limit.md` | HC `_up` FP8 −3.6 % at c=1 | n=1, signs alternate | **UNDER-SAMPLED** (mechanism supported by shapebench; the number is not) |
| same | 128-divisibility is a class constant | code readout | **EXISTENCE PROOF** |
| `skinny-gemm-on-sm121.md` | sm_103 configs buy nothing (−1.5 % ± 1.37) | n=6 | **ADEQUATELY SAMPLED** |
| `why-the-hyper-connections-do-not-respond.md` | HCs latency-bound at ~78 % of roofline | shapebench + 3 interventions at n=6 + field | **ADEQUATELY SAMPLED** |
| same | larger cudagraph capture sizes null | n=1 | **UNDER-SAMPLED** |
| `fp8-kv.md` | ×1.72 KV pool; decode +2.6 % no regression | config readout; n=6 per arm | **ADEQUATELY SAMPLED** |
| same | NIAH 5/5 as quality gate | 5 depths | **vacuous** — `TODO.md`: "discriminates nothing"; also backs the ring-widening safety claim in README and the PR body |
| `cudagraph-mode.md` | `FULL_DECODE_ONLY` +17 % KV pool | readout | **ADEQUATELY SAMPLED** |
| same | −2 % latency (42.78) | n=1, = the band minimum | **UNDER-SAMPLED**, self-flagged |
| `prefill-batch-size.md` | 2048 costs ~13 % deep TTFT; 16384 buys nothing | n=3 | **ADEQUATELY SAMPLED** |
| same | 16384 costs ~4.5 % per agent turn | n=2 vs n=3 inside an 11 % band | **UNDER-SAMPLED** |
| `moe-backend-axis.md` | quantized drafter: no measurable decode cost | n not stated | **UNDER-SAMPLED** (null; state n) |
| `fetching-a-slice.md` | 202 of 206 shards byte-identical | sha256 census | **ADEQUATELY SAMPLED** |
| `choosing-a-quant-scheme.md` | FP8 error ~2.25 % is the E4M3 floor | 6 tensors × 3 schemes | **ADEQUATELY SAMPLED** |
| `failure-modes.md` A1–B2b | defect signatures | one occurrence + hash/code each | **EXISTENCE PROOF** |

## 4. Profile / structural / capability claims

| file | claim | evidence | class |
|---|---|---|---|
| `single-stream-limit.md` | GPU 95.5 % busy; 66.7 % in cuBLAS BF16 GEMV | one trace (253,529 intervals) + bandwidth floor | **ADEQUATELY SAMPLED** (census, two methods) |
| `where-the-gpu-time-goes.md` | HCs are 25 % of GPU time | trace + call-count arithmetic | **ADEQUATELY SAMPLED** |
| same | quantizing the drafter's experts worth ~5 % | derived, never measured e2e | **UNDER-SAMPLED**; `moe-backend-axis.md` saw no cost |
| `the-prefill-decode-confound.md` | profile was prefill-contaminated | re-profile same ranking | **WITHDRAWN** in place |
| `load-and-waits.md` | 266.8 tok/s at c=48; majflt/token falls 4.4× | n=1 per cell | **UNDER-SAMPLED** for levels; shape is one run |
| same | PLE offload not the bottleneck | /proc counters, 12 cells | **ADEQUATELY SAMPLED** |
| `ple-access-pattern.md` | 16 rows / 2.5 KB per token; ≤5 % budget | arithmetic + n=1 majflt | arithmetic **EXISTENCE PROOF**; levels **UNDER-SAMPLED** |
| `batch-invariance-unavailable.md` | structurally unavailable | code + RuntimeError | **EXISTENCE PROOF** |
| `tool-calling-was-off.md` | tool calling 32/32 | 8 trials × 4 temps | **ADEQUATE** for "works"; bounds failure at ≲9 %, not 0 |
| `temp0-nondeterminism.md` | not reproducible from ~30 tokens; single-variable exclusions | 5 reqs × cells | **EXISTENCE PROOF** |
| `read-only-is-not-load-free.md` | subagents inflated 4 MTP arms by 24–83 % | 9 MTP arms, n=1 each | **UNDER-SAMPLED and now confounded** by the per-request draw |

## 5. Contradicted by later findings, not corrected in place

| # | sentence | where | contradicted by |
|---|---|---|---|
| 1 | "k=2 is confirmed the MTP optimum" | `which-drafter-for-agent-work.md`:39 | `README.md`:118-120 withdraws it; `mtp-depth-anomaly.md`:29 |
| 2 | "k=2 as the optimum is confirmed three independent ways" | `why-the-hyper-connections-do-not-respond.md`:115 | same; the k=3 leg is the retracted n=1 arm |
| 3 | "k=2 remains optimal on this build" | `fp8-mixed-checkpoint.md`:118 | same |
| 4 | TRTLLM/CUTEDSL/VLLM_CUTLASS/MARLIN/HUMMING "untried" | `moe-backend-axis.md`:99; `the-field.md`:539 | finding 28 measured marlin, humming, cutlass |
| 5 | "MoE routing remains the leading structural candidate" | `temp0-nondeterminism.md`:78 | finding 26: routing excluded; the experts kernel is the cause |
| 6 | `persistent_topk` "explains every observation" | `temp0-nondeterminism.md`:123-164 | findings 5 and 26 |
| 7 | "n=6 is genuinely 2.3× slower" | `mtp-depth-anomaly.md`:113 | `failure-modes.md`:607 "there is no n=6 anomaly" |
| 8 | "every arm inside an agent window is inflated 24–83 %" | `read-only-is-not-load-free.md`:21 | finding 41: 12 quiet starts span 33–70 ms/tok |
| 9 | "divergence comes from the machinery being active" | `prefix-cache-is-not-reuse.md`:31 | finding 27 |
| 10 | "A single forward pass is deterministic" | `determinism-investigation.md`:7; `prefill-divergence.md`:7 | finding 27 (M=52 and M=3 diverge) |
| 11 | "median n=5 would move 58.4 toward ~32" | `prefill-divergence.md`:175 | finding 40 |
| 12 | `pidfd_getfd` denied by Docker seccomp | `results-radixark-vllm.md`:82,90 | `failure-modes.md` B2: it is `ptrace_scope=1`, bare metal too |
| 13 | "k=5 hard-fails" | `REPRODUCE.md`:119; `TODO.md`:92 | widening patch applied; vllm#54912 |
| 14 | "Never combine MTP with --async-scheduling" | `REPRODUCE.md`:121 | `upstream-branch.md`:96: async is on by default with MTP; every MTP number here ran with it on |
| 15 | prefill "flat 2003–2380" | `REPRODUCE.md`:154 | `REPRODUCE.md`:158 itself: ±20 %, 1633–2367 for one config |
| 16 | "n=5 is the best at 31.9" | `README.md`:120 | `mtp-depth-anomaly.md`:22: n=5 over 17 arms min 31.8 / median 58.4 — the README quotes the minimum |
| 17 | ngram_gpu fastest edit-shaped (n=1); ngram advantage grows (n=1) in the "Provisional" list; "acceptance not instrumented" | `evidence-standard.md`:52-59 | retired at n=2 in `which-drafter…`:59; acceptance instrumented since finding 8 |
| 18 | three different k=4 summaries (~35 n=2; 31.7 n=2; min 31.5/median 35.4 n=4) | `evidence-standard.md`:55; `mtp-depth-anomaly.md`:21,157 | internal disagreement |
| 19 | "MTP k=2 n=1 clean" vs "k=2 48.6, 3 arms" | `which-drafter…`:83 vs `mtp-depth-anomaly.md`:54 | internal disagreement on the headline drafter number |

## 6. Documents whose headline verdict depends on an under-sampled MTP number

| document | headline | rests on | n | status |
|---|---|---|---|---|
| `depth-curve.md` | MTP ~1.54× that does not erode with depth; "stays on at every depth" | one start for all 5 depths | 1 | **fails** — a second start landing broken reverses the sign |
| `mtp-depth-anomaly.md` | k=2 never reaches the fast regime; run k=3/4/n=5 | 1–4 starts/config, no per-turn acceptance | 1–4 | **fails** (≈40 % chance by the draw alone) |
| `mtp-vs-prefix-cache.md` | run MTP k=2 above ~68 output tokens | 30- and 400-tok cells | 1 | threshold **fails**; the 1600-token cache cost (3 starts, counter) survives |
| `speculation-on-flash-next.md` | +44 % decode; TTFT −30 % "the reason to keep MTP on" | 1 start/arm | 1 | **fails**; contradicted at higher n by two later notes; README still links it as authority |
| `which-drafter-for-agent-work.md` | agent work → ngram n=4; k=2 the MTP optimum | ngram n=2, k=2 n=1 | 1–3 | ngram direction survives; every MTP row fails |
| `single-stream-limit.md` | two-thirds of decode is BF16 GEMV | trace census | — | **does not depend**; only its appended MTP table (+67 %) is under-sampled |
| `results-radixark-vllm.md` | coherent, correct, concurrent | no-spec, ≥3 measurements | ≥3 | does not depend on MTP; "near-linear concurrency" is 2 points; Docker attribution superseded |
| `fp8-mixed-checkpoint.md` | +71 % single-stream | 17.1→29.2 (MTP k=2, n=1) | 1 | **partly fails**: +39 % no-spec half is adequate; +71 %, complements table, "k=3 not optimal" are n=1 MTP cells |
| `prefill-divergence.md` | forward pass not deterministic (corrected) | 1 arm | 1 | tail prediction refuted by finding 40, unamended |
| `read-only-is-not-load-free.md` | subagents invalidated five arms | 9 MTP arms | 1 | **fails** — confounded with the per-request draw; the rule is worth keeping, the attribution is not |
| `upstream-report-draft.md`, `upstream/issue-vllm-moe-fused-finalize.md` | nondeterminism fixed at +3.6 % | 45.50 vs 43.92 | 1 vs 1 | determinism claim is the repo's best-evidenced result (finding 37); the **cost number is n=1 against an 11 % band** and appears four times |
| `README.md` / `REPRODUCE.md` | 36.5 tok/s via three levers | n=6, sd 2.9 % | 6 | the 36.5 is adequate; `README.md`:120 and `REPRODUCE.md`:119/121/154 are not (§5 #13–16) |
| `cudagraph-mode.md` | ship `FULL_DECODE_ONLY` | −2 % (n=1) + 17 % KV pool | 1 | survives on the KV pool alone |

## 7. The three runs that retire the most

1. **3 starts × 40 turns, MTP k=2, per-turn acceptance** — anchors `which-drafter…`, `mtp-depth-anomaly.md`, the "Solid" list and the README's k=2 withdrawal; the only MTP depth with zero per-turn acceptance data.
2. **3 starts × 5 depths, MTP on/off** — retires the "+52–54 % flat to 60k" headline and "MTP stays on at every depth".
3. **3 interleaved no-spec arms each of `flashinfer_cutlass` / `use_fused_finalize=False` / `emulation`** — turns the +3.6 % and +17 % cost numbers (both n=1, both drafted for upstream) into something the 42.8–47.7 band cannot swallow; add the two missing shapes for the emulation determinism claim.
