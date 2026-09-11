# Determinism investigation — state of play (2026-09-01)

Index for the night's work. Detail lives in the linked notes; this is what is established, at the
confidence each item deserves, plus what was refuted along the way.

## Answers by question — read this before re-measuring anything

This file is chronological and long. It answers "what happened on the 4th" well and "what does X
cost" badly, which is the question people actually arrive with. Added 2026-09-08 after three
subagents *and I* re-asked a question this file had answered twice (det-168). **Keep it current:
a stale answer here is worse than no answer, because this is the part people trust.** Last
reconciled against the findings at det-191.

| question | answer | findings |
| --- | --- | --- |
| Is the QSA top-k kernel deterministic? | Stock: no (0/4 prompts, 0/81 shapes). Our #55122: yes, 81/81, and index-canonical. | 76, det-151, det-165 |
| **What does the deterministic kernel cost end to end?** | **Nothing measurable.** TTFT and s/turn inside the start-to-start band over 3 starts/arm; independently, 99–100 % of stock prefill throughput. (87–90 % is the *Python* fallback, not the kernel.) | **82, 76** |
| What does it cost per call, in the microbenchmark? | **0.72–1.78× over 43 cells** since the blocked emission + `RADIX_THRESHOLD` 22,016; at or below stock on 27 of them. Was 1.01–2.13× before those. | det-171, det-172, **det-173** |
| Which cell is worst, and can it be fixed? | 64 × 24,576, at 1.78×. It sits just above the caching bound (n ≤ 24,280) so it is multi-CTA under *every* legal threshold — that constant cannot reach it. | det-172, det-173 |
| Is the opt-in FlashInfer backend (#55872) an alternative on GB10? | **No — it does not start on sm_121**: `TopKRaggedTransform … operation not supported` at engine init. Their patch is fine; the kernel has no sm_121 path. | det-175 |
| Does prefix caching work under MTP, despite the "reuse will be disabled" warning? | **Yes** — 92.1 % hit rate on repeat requests, exactly 4 full blocks. The warning's operative clause is about an external KV offload tier, which we do not run. | det-174 |
| Does it change output quality? | Third-party 50-item suite: 95/100 with 0/50 unstable, vs stock 97/100 with 13/50 unstable. | 76 |
| Does it give batch invariance? | No, and it cannot — GDN has no batch-invariant path. | 76 |
| Is #55314 an alternative? | No. It fixes the set nearly for free but not the order, and its tie clips make the set scheduling-dependent. | det-165, det-166, det-167 |
| Is a merged #55122+#55314 kernel worth building? | No — closed, not deferred, on the end-to-end number above. | det-166, det-167, det-168 |
| Is ZC502's position-parity collector usable here? | The **client** one is: 12/12 runs on sm_121. The offline one is not (it constructs `LLM()` in-process). | det-155, det-178 |
| **Does greedy decoding actually diverge end to end on TRUE stock?** | **Yes, badly.** 335 disagreeing positions on a 2.5k-token prompt, forced-logprob spread 10.63, first divergence at position 2, 104 modal top-1 mismatches. With all four fixes: exactly 0. Eight earlier "nulls" had three of the four silently active. | **det-181**, det-180 |
| Which of the four fixes carries that? | **No single one — they are jointly necessary.** Isolated: none 333, qsadet 330, cachekey 334, plefix 285, detfin+cachekey 280, **all four 0**. And `plefix` is *not upstream* (det-182), so our own #55122 is **not** the load-bearing fix; say so upstream. | **det-184** |
| Is the nondeterminism the top-k SELECTION or the SCORES fed to it? | **The scores.** Across 7 identical requests on stock, all 13 comparable prefill calls have differing input scores; **zero** have identical scores with a differing selection. `all4` is bit-identical on both. Answers @rybruscoe's discriminator on #54521. | **det-191** |
| Why does `qsadet` alone do nothing? | Because the defect is unreachable on this traffic: of 6,192 rows that actually performed a top-k selection, **0 had any tie at the k-th value** (and 93 % of rows had fewer visible blocks than k, so selection was a no-op). #55122 is kernel correctness under ties, not end-to-end determinism here. | **det-190** |
| Is the PLE offload subsystem upstream? | **No.** Zero `vllm/v1/ple_offload/` files in the dev524 wheel and 404 on vllm main; vllm#53899 is open and `mergeable_state: dirty`; our semaphore fix (PR #13 on its fork branch) is open. It exists only here and on that branch. | **det-182** |

## Established (measured, replicated where stated)

1. **A single forward pass is deterministic.** With `--no-enable-prefix-caching`, three identical
   single-token requests return bit-identical logits (`lp=-0.2308074981`, `sig=e22e0de36cac`, 3x).
   This *reverses* the working assumption held for most of the session.
   → `prefill-divergence.md`

2. **The divergence happens at 0.0% prefix-cache hit rate.** The probe prompt is ~60 tokens against
   a **1616-token** block, so no block ever completes and nothing is ever reused; sub-block hits
   need `--prefix-match-unit`, which we never set. The same build hits **55.3%** on the 8-turn
   agent loop, so the cache is not broken — the probe simply cannot hit it. **Therefore the cause
   is the align-mode state machinery running, not cached state being reused.** This kills every
   "stale/wrong/truncated checkpoint" hypothesis for this probe.
   → `prefix-cache-is-not-reuse.md`

3. **Generation diverges independently.** With the prefix cache off, a 1040-token generation still
   gives 3 distinct outputs of 3. So there are **two** paths, not one. Not established: whether
   they share a cause.

4. **Divergence enters at layer 1** — with an important caveat added later by the sub-bisection:
   the fingerprint used to group passes (a layer's output tensor) is **too weak on this
   architecture**, because hyperconnections carry multiple residual streams between layers. Layer
   1's `in_proj_qkvz` differs while `layers.0`'s output is identical, and a plain GEMM cannot do
   that unless its input differs. So read this as "the first HOOKED module that differs", not
   "where divergence starts". Everything PLE-table-derived is identical; everything
   hidden-state-derived differs. See `prefill-divergence.md`.

   *(original finding as measured)*

4b. **Divergence enters at layer 1.** Hashing every decoder layer over repeated identical prefills,
   with passes grouped by their **layer-0 hash** so like meets like: `layers.0` identical x3,
   `layers.1.ple.ple_embedding` identical x3, `layers.1` differs. Replicated in two independent
   groups of three passes. Layer 1 is the only layer carrying the PLE; layers 0-2 are all
   `linear_attention`, so attention type is not the difference.

5. *(QSA top-k: superseded by finding 54 — the exclusion held only while the MoE masked it.)* **Excluded as causes:** speculation (`P_nospec`, `G_eager_nospec`), CUDA-graph replay
   (`G_eager_*`, verified 0 captures), the QSA top-k (`torch.topk` substitution still diverged),
   PLE FP8 quantisation (0 dead rows in 2100 sampled, 2.9x total dynamic range), missing Triton
   bounds masks (every unmasked access is bounded by construction), and the top-k -> block
   expansion interface (the consumer re-derives the bound from `sequence_length`).

6. **MTP throughput is not reproducible across restarts** — up to **1.83x** within one config,
   while no-spec (1.10x) and ngram (1.09x) are stable. This killed the "MTP n=6 anomaly", which
   never existed. `k=2 slower than no speculation` survived and got *stronger* (4 arms >= 47.8 vs
   12 arms <= 47.7, non-overlapping).
   → `mtp-depth-anomaly.md`, `failure-modes.md`

7. **`VLLM_BATCH_INVARIANT` cannot run on this architecture.** No mamba/linear-attention backend
   implements `supports_batch_invariance()`, and 36 of 48 layers are linear attention. The one
   off-the-shelf remedy is unavailable, and cannot be offered to anyone running this model class.
   → `batch-invariance-unavailable.md`

8. **Acceptance is the channel that turns divergence into the throughput spread.** Five starts
   of one config (MTP n=5), pairing `ms/tok` with `mean_accept_len`:

   | arm | ms/tok | accept rate | mean_accept_len | draft work kept |
   | --- | ---: | ---: | ---: | ---: |
   | AC4 | 32.16 | 68.9% | 4.45 | 89% |
   | AC1 | 32.83 | 63.2% | 4.16 | 83% |
   | AC5 | 48.90 | 24.2% | 2.21 | 44% |
   | AC2 | 62.75 | 11.3% | 1.57 | 31% |
   | AC3 | 71.73 | 7.1% | 1.35 | 27% |

   **pearson r = -0.964.** Same binary, same flags, same prompts: MTP keeps either ~89% or ~27%
   of its draft work, and the no-spec reference is 43.5 ms/tok — so the collapsed arms are
   **worse than not speculating at all** while the healthy ones are ~26% better.

9. **The regime is per-TURN — and, CORRECTED, it is NOT one-way early on.** A 40-turn run
   (`DEG_a`, per-turn acceptance deltas, raw table in `notes/data/DEG_a-per-turn.txt`) shows:

   - **Turns 1-27: a per-turn coin flip.** Acceptance alternates between ~4.5 (healthy) and ~1.4
     (collapsed) and **recovers repeatedly** — turns 3, 10, 13, 16, 18, 23 all bounce back after
     a collapsed turn. The earlier "no arm recovers" claim came from 8-turn windows too short to
     see a recovery; it is withdrawn.
   - **Turn 28 onward: locked at 1.2-2.1 for 13 consecutive turns.** That part IS one-way.
   - **Every attention-block-boundary crossing (1616 tokens) coincides with a collapsed turn,
     3 of 3**: prompt crosses 5x1616 at turn 5 (2.41), 6x1616 at turn 17 (1.59), 7x1616 at
     turn 28 (2.08, then lock-in). Collapses also occur off-boundary (turns 2, 8, 9, 11, 12, 15,
     22), so a boundary is *a* trigger, not the only one.
   - Timing follows: collapsed turns ~7-9 s, healthy turns ~3-4 s.

   **DEG_b (n=2) — two of the three claims above did not survive it:**

   | | DEG_a | DEG_b |
   | --- | --- | --- |
   | block crossings on a collapsed turn | 4 of 4 | **1 of 4** (turns 6, 18, 28 healthy at 4.26 / 3.82 / 4.48) |
   | per-turn distribution (healthy / collapsed / mid) | 18 / 22 / 0 | 21 / 16 / 3 |
   | recoveries | 7 | 7 |
   | last healthy turn | 27 | 37 |

   - **Replicated: the per-turn coin flip.** Sharply bimodal, almost no mid values, ~40-55% of
     turns collapsed, seven recoveries each. This is the robust finding.
   - **Refuted: block-boundary triggering.** DEG_a's 4-of-4 was coincidence.
   - **Lock-in weakened to n=1.** DEG_a's 13 consecutive collapsed turns is p~1e-4 under a fair
     coin, so probably real; DEG_b's last healthy turn is 37, leaving 3 collapsed at the end,
     which is what chance looks like. Not replicated within 40 turns.

   **DEG_c (n=3) — the three-arm picture:**

   | arm | healthy | collapsed | collapse rate | recoveries | longest collapsed run | crossings collapsed |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | DEG_a | 18 | 22 | **55%** | 7 | 13 | 4/4 |
   | DEG_b | 21 | 16 | **40%** | 7 | 3 | 1/4 |
   | DEG_c | 32 | 5 | **12%** | 4 | 2 | 0/4 |

   - **Replicated 3/3: the per-turn bimodal flip, with recovery every time.**
   - **New: the flip's BIAS is per-start** — 12% to 55% collapse rate across identical starts. So a
     per-start component exists on top of the per-turn flips. This is exactly why 8-turn arms
     spread 1.83x: they sample both.
   - **Block boundaries refuted**: 5 of 12 crossings collapsed vs a 36% base rate.
   - **Lock-in unresolved, 1 of 3.** DEG_a's 13-run is p~4e-4 even at its 55% bias, but three
     40-turn arms are many windows. Not established; not excluded.

   **DEG_nospec — the control is FLAT; the acceptance mechanism holds.** Per-turn seconds,
   turn 1 dropped (cold prefill):

   | arm | mean s/turn | max/min | CV | <5 s | >6.5 s | between |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: |
   | DEG_a (MTP) | 6.14 | 3.10x | 0.37 | 17 | 19 | 3 |
   | DEG_b (MTP) | 5.60 | 2.86x | 0.38 | 22 | 13 | 4 |
   | DEG_c (MTP) | 4.30 | 2.67x | 0.29 | 33 | 3 | 3 |
   | **DEG_nospec** | **5.32** | **1.29x** | **0.05** | 0 | 1 | **38** |

   Without a drafter, 38 of 39 turns sit in 5.06-5.56 s (the outlier is turn 2's warm-up tail);
   the MTP arms are bimodal with almost nothing in the middle. **The flip lives entirely in
   speculative acceptance; the base decode path is steady.** This retires the alternative reading
   that something degrades regardless of the drafter.

   Raw tables: `notes/data/DEG_{a,b,c,nospec}-per-turn.txt`.

   (earlier n=1 text:) **n=1.** Whether the lock-in position (~7th block) replicates is what `DEG_b`/`DEG_c` decide;
   `DEG_nospec` is the control that must stay flat.

   *(earlier text, from 8-turn arms, kept for the record)*

9-old. **The regime is per-TURN, not per-start, and the transition is one-way.** Per-turn timings,
   comparing turns 2-4 against 6-8 within each run:

   | arm | early | late | ratio |
   | --- | ---: | ---: | ---: |
   | AC1 | 3.96 | 4.03 | 1.02x (healthy throughout) |
   | AC4 | 3.87 | 3.68 | 0.95x (healthy throughout) |
   | AC2 | 7.68 | 6.82 | 0.89x (degraded from turn 1) |
   | AC3 | 9.35 | 8.39 | 0.90x (degraded from turn 1) |
   | **AC5** | **4.14** | **8.17** | **1.97x — transitioned mid-run** |

   AC5 ran fast for turns 2-4 (4.30/3.70/4.43) then slow for 6-8 (5.08/10.22/9.21); its
   "intermediate" aggregate is an artifact of averaging two regimes. **No arm recovers.** A
   one-way transition is what you expect if a rejection leaves recurrent state that cannot be
   rewound — the exact root cause named in the unmerged half of vllm#47861.

   ⚠️ Corrected twice while measuring: bimodality claimed at n=4, apparently refuted by AC5's
   intermediate aggregate, then restored at per-turn granularity. The unit of analysis was the
   error, not the data.

10. **H1 is dead for the prefill divergence: a forced sync after the align postprocess does not
    restore determinism.** `SYNC` arm (cache on, `VLLM_ALIGN_SYNC=1`, `torch.cuda.synchronize()`
    inserted after `postprocess_mamba_align_gpu`): 3 distinct of 3, lp -0.247 / -0.400 / -0.595.
    Env delivery verified: the `$12` slot builds one argv entry by the same path every `FN_*`
    variable takes, and those are honoured in the same log. Caveat kept honest: the patch had no
    log line, so "fired" rests on the delivery mechanism, not a positive trace.

11. **The `'#'` vs `'The'` top-token split was MY sort hook, not the kernels.** On the cleaned venv
    (sort removed, stock `persistent_topk`), `SYNC` gives `'#'` in the -0.25..-0.59 band — the same
    band as every `torch.topk` arm. Only arms with `persistent_topk` **plus the sort** gave `'The'`
    at -1.51. Mechanism: when `visible < k` (the 60-token probe: ~15 of 512 slots), the kernel
    fills slots >= visible with **-1** (corrected by the oracle test, finding 18 — not `torch.empty`
    garbage); sorting the whole row moved those -1s below the real indices, where the expansion kernel reads them. `torch.topk`'s
    padding is all >= visible, so it was immune. T5/T6 in `topk_boundary.py` will confirm.

    **Which earlier arms are affected**: every underfilled probe that ran `persistent_topk` with
    the sort — `NOPFX_a/b/c`, `BISECT`, `BISECT2` (the `'The'` / -1.x band). NOT affected: arms
    with `torch.topk` (`P_ctl`, `P_nospec`, `F_noprefix`, `G_*`), and NOT the agent-loop /
    `degrade` arms (an 8k+ prompt gives ~2000 visible blocks > k=512, so no padding exists).

    **Which conclusions survive**: all the categorical ones. "1 distinct of 3" is still
    deterministic and "3 of 3" still diverges whatever the token; the layer-1 bisection is still
    a divergence. What is retracted is any *value* quoted from a sort-affected arm — the -1.51
    logprob, the `'The'` token, and finding 2's "top-k changes the answer", which is withdrawn.

12. **Async scheduling is not the mechanism either.** `NOASYNC` (cache on, `--no-async-scheduling`,
    log evidence: `(APIServer pid=275763) INFO 09-01 23:51:00 [api_utils.py:272] non-default args: {'model_tag': '/opt/llm/models`): 3 distinct of 3. So neither the postprocess sync
    (finding 10) nor removing the second in-flight batch restores prefill determinism.
    `LAUNCHBLOCK` (`CUDA_LAUNCH_BLOCKING=1`) is the remaining race test: if it also diverges, the
    prefill source is not a launch-ordering race at all but a deterministic-yet-scheduler- or
    state-dependent path.

13. **The prefill divergence is NOT a race.** `LAUNCHBLOCK` (cache on, `CUDA_LAUNCH_BLOCKING=1`,
    delivered via the same `$12` slot verified for `SYNC` — that mechanism is the sole basis; an
    earlier "load visibly slower" corroboration was checked and withdrawn: arm spacing was
    13/14/13 min, no slowdown, which is expected since launch blocking slows execution, not
    I/O-bound loading): 3 distinct of 3. With findings 10 and 12 that closes the race family — postprocess
    sync, single batch in flight, serialised launches — all three leave it diverging. **The source
    is a deterministic-but-state-dependent path**: something the align machinery does that depends
    on state carried between requests, not on timing. This fits the per-start bias (finding 9,
    12-55%) far better than any race did, since a race gives the same bias every start.

    Oracle test note: `persistent_topk` accepts only k in {512, 1024, 2048}; the test's default
    k=16 was wrong and it is re-queued at the production k=512 (`oracle` unit, after `rerun`).

14. ~~**Cache off + speculation off: a 4-token generation is DETERMINISTIC**~~ **WITHDRAWN, see 20** (`GENBIS`, eager,
    identical signature x3, every prefill pass identical at every hooked module). The only
    generation-side divergence measured with the cache off (`F_noprefix` NOFB, 1040 tokens) had
    speculation ON, so the generation-side source may be spec-dependent (H3) rather than a
    separate base-path defect. Still to separate: length (4 vs 1040) from speculation.

15. ~~**Instrument flaw #2: the layer hook hashes non-semantic rows in decode.**~~ **WITHDRAWN, see 20 — the hook was right, the probe was blind** In `GENBIS` the
    first decode pass of each request — same layer-0 hash, same token, same state — shows
    layers 1+ DIFFERING while the OUTPUT is identical. Real differences cannot vanish before the
    logits, so the hook is hashing padded/stale rows beyond the single real token.
    **Consequence:** "first differing layer" from any DECODE pass is unsafe. Prefill-pass claims
    (`BISECT`, where the output also diverged) still show that divergence exists, but "first at
    layer 1" is only as good as the hook. Fix built (v3: row0 hash + shape) and queued as
    `GENBIS2` after the oracle; prediction: row0 identical, full differs.

16. **`mamba_cache_mode=all` diverges exactly like `align`** — `M_all_a/b/c`, cache on, mode
    verified in the log (`mamba_cache_mode': 'all'`): 3 distinct of 3 in all three arms. As
    vllm#54173 reported and finding 13 implied, the state-dependent path is in machinery
    **common to both checkpoint policies**, not in `align`'s last-token-of-scheduler-step rule.
    `M_align_*` follows as the explicit control.

    **Scatter, not offset.** On the cleaned venv the cache-off reference (`GENBIS`, -0.2566) sits
    inside the cache-on range (`SYNC` -0.247..-0.595, `M_all` -0.345..-0.756, all `'#'`). So
    cache-on is *nondeterministic around* the uncached answer rather than *systematically
    displaced* from it — the reproducibility defect, not (on this probe) a quality offset.
    Caveat: `GENBIS` is eager + spec-off, the cache-on arms are graphs + MTP; the exactly-matched
    cache-off reference (`NOPFX_*`) was sort-corrupted and must be re-run clean before this is
    quoted as more than indicative.

17. **`mamba_cache_mode` is irrelevant to the divergence.** `M_align_a/b/c` (the explicit control):
    3 distinct of 3 in all three. With finding 16 that is **6 of 6 arms diverging across both
    checkpoint policies**. The state-dependent path is in machinery common to both.

18. **Oracle test at the production k=512 (`persistent_topk` on sm_121):**
    - **T1-T4: no failure at any of 11 swept sizes >= 512** (512, 513, 1023-1025, 4095-4097,
      8447-8449) — no read past `visible_blocks`, first and last valid index selected, and the
      selected SET matches `torch.topk`. Two caveats: the test printed only failures, so this is a
      silence-is-pass result (fixed: it now prints per-size PASS); and a set comparison cannot see
      the *value* error vllm#54521 reports at ~8448, so that is neither confirmed nor refuted.
    - **T5 corrects my mechanism**: the kernel does NOT leave padding as `torch.empty` garbage — it
      writes **-1** into every slot >= visible (0 untouched of 3976/3616/2496). The sort hook then
      moved those -1s *below* the real indices, so the expansion kernel read block index -1 for the
      first `visible` slots. That is **deterministic** garbage, which is exactly why `NOPFX_*` was
      stable-but-wrong (`'The'`, -1.51, six identical requests): a deterministic corruption, not a
      random one. Finding 11 stands; its "torch.empty garbage" wording is replaced by "-1 fill".
    - T6 (simulated garbage + sort) fails at vis 15/60/200 as predicted; the real case is simpler
      and worse than the simulation.

19. **The CUDA/PyTorch runtime is not the source.** `runtime_determinism.py`, idle GPU: bf16 GEMM
    (4096x2560x10240 and 1-row decode shape), fp32 8192^2 reduction, SDPA 24h x 4096, top-k with
    ties — **all bit-identical over 5 runs, including with allocator churn between runs.** A bound,
    not a proof (5 runs), but consistent with cross-start determinism (finding on `NOPFX`): the
    divergence lives above the runtime, in vLLM's own kernels or model code.

20. **Findings 14 and 15 are WITHDRAWN — the probe was blind to decode.** `logitprobe.py` hashed
    only `lp[0]`, the first token's top-k. With `max_tokens=4`, tokens 2-4 were never checked, so
    `GENBIS`/`GENBIS2` "1 distinct of 3" meant only "prefill deterministic" (already known). The
    layer hook was fine: `GENBIS2` v3 shows decode layer-0 output shape `(1, 10240)` — one row, no
    padding — and **row 0 differs at 47/48 modules**. Not a padding artifact.

    **What the hashes actually establish: with the prefix cache OFF and speculation OFF, decode
    diverges from layer 1 at the very first decode step**, while the prefill feeding it is
    bit-identical at every module (`GENBIS2` prefill groups: 0/48 differ). So the generation-side
    source is NOT spec-dependent and NOT the prefix cache. Between the identical prefill and the
    diverging first decode step, what changes is (a) the GDN recurrent state handed from prefill
    to decode — a state cache that exists whether or not prefix caching is on — and (b) the
    decode-shaped kernels. Whether the divergence reaches the sampled tokens within 4 steps is
    unmeasured (the fixed probe now hashes every token); over 1040 tokens it does (`F_noprefix`
    NOFB). Re-run queued as `GENBIS3`.

21. **With the prefix cache OFF and speculation OFF, the first decode step's OUTPUT already
    differs** (`GENBIS3`, eager, fixed probe hashing every token):

    | token | req1 | req2 | req3 |
    | --- | --- | --- | --- |
    | 1 (prefill) | `355fed8e` | `355fed8e` | `355fed8e` |
    | 2 (decode step 1) | `8052946b` | `9e428fac` | `ace4724f` |
    | 3, 4 | differ | differ | differ |

    This is the output-side confirmation of finding 20's layer hashes: prefill bit-identical, decode
    divergent from step 1. It also means the `F_noprefix` NOFB divergence was never about length or
    speculation — decode diverges immediately, with neither. **Both divergences now sit on the
    recurrent-state path**: cache-on prefill (align machinery writing/reading state) and cache-off
    decode (the state handed from prefill to decode step 1). Next single-flag test:
    `VLLM_GDN_DECODE_KERNEL=cuda` vs the `triton` we run (chosen because cuda hung at c~32; c=1
    is safe) — separates "the decode kernel" from "the state handoff".

22. **`GDNCUDA_a` was a triton replicate, not the cuda test.** Its log reads `GDN decode kernel:
    triton`; the runner lineage it came from never had the `$12` env slot, so
    `VLLM_GDN_DECODE_KERNEL=cuda` was silently dropped as an extra positional. Delivery itself was
    re-verified end-to-end on a non-GPU unit (two `Environment=` properties arrive intact and the
    launcher's `${VAR:-triton}` keeps `cuda`). `quick.sh` did have the slot, so `SYNC`'s
    `VLLM_ALIGN_SYNC=1` was delivered and finding 10 stands. As a replicate it is still useful:
    **n=2 for "cache off + spec off: prefill identical, decode step 1 diverges"** (finding 21).
    Slot wired; the cuda arms re-launched. Rule added to the runner: a `$12` value must show up in
    the arm's log or the arm does not count.

23. **The GDN decode kernel choice moves WHERE divergence first appears — it does not remove it.**
    `GDNCUDA_a` (cache off, spec off, eager, `GDN decode kernel: cuda` confirmed in the log;
    prefill kernel unchanged, Triton/FLA):

    | decode kernel | token 1 (prefill) | token 2 (decode step 1) |
    | --- | --- | --- |
    | triton (`GENBIS3`, old `GDNCUDA_a`; n=2) | **identical** x3 | differs |
    | cuda (`GDNCUDA_a`, `_b`; **n=2**, kernel line verified `cuda` in both logs) | **differs** x3 in both arms | differs |

    **Why a "decode" flag changed prefill**: `use_fused_gdn_decode` (`qwen_gdn_linear_attn.py:892`)
    is gated only on `enable_fused_gdn_decode` and dtypes — **no decode-step check** — so with
    `cuda` the fused packed op (`qwen_gdn_attention_core_fused_norm_packed`) runs for EVERY forward,
    prefill included. The cuda arm is therefore a *second GDN implementation throughout*, not the
    same prefill with a different decode. Two independent implementations, both nondeterministic
    on the same state path (`state=self.kv_cache[1]`, read/written in place). Neither is "the source": both variants lose determinism on the GDN recurrent-state
    path, cuda one step earlier. **The state write at the end of prefill and its read at decode
    step 1 are now the narrowest suspect**, and the next instrument is to hash the GDN state
    tensors directly at those two points.

24. **THE STATE HASHES SPLIT IT: layer 0 is deterministic through decode; layer 1 diverges at
    decode step 1; the only structural difference is the PLE.** `STATEHASH` arm (cache off, spec
    off, eager, triton; hook hashes conv+SSM state at the request's slot after each GDN forward;
    raw in `notes/data/STATEHASH-run.txt`):

    | | prefill state | decode step 1 | step 2 | step 3 |
    | --- | --- | --- | --- | --- |
    | layer 0 (GDN) | identical x3 | **identical** | identical | identical |
    | layer 1 (GDN + PLE) | identical x3 | **DIFFERS** | differs | differs |

    - **The state write at the end of prefill (`qwen_gdn_linear_attn.py:1520`) is deterministic**
      in both layers — the earlier "state handoff" suspicion is narrowed away from the write.
    - **Layer 0's decode is fully deterministic**: same GDN kernel, same state shape
      `(48,128,128)`, same slot ids in play, three requests, three decode steps, all identical.
      So the triton decode kernel is not nondeterministic on its own.
    - **Layer 1 diverges at decode step 1** on a state that was identical one step earlier. Layer 1
      is layer 0 plus the PLE (`nvidia/model.py:296`, `hidden_states = hidden_states + self.ple(…)`),
      which runs in a **CPU-offload subprocess with an async connector** — the very component
      hand-patched for an event-pool race (`4e8b849`).
    - Consistent with the very first bisection ("layer 1 first differs"): there the PLE *lookup*
      hashed identical, but only in **prefill** passes; the PLE's **decode-step** output was never
      hashed. The sampled token feeding decode step 1 is identical (token 1 `355fed8e` x3), so the
      n-gram context is identical too — a differing PLE output at step 1 would be the offload
      machinery, not the model.

    **Next, and decisive**: hash the PLE's return value and layer 1's GDN input at decode step 1
    (with the double-hash race detector), across three identical requests.

25. **The PLE is exonerated; the divergence enters between layer 0's return and the PLE's input,
    in the hyperconnection's DEFERRED block output.** `PLEHASH` arm (cache off, spec off, eager,
    triton; raw in `notes/data/PLEHASH-run.txt`), across three identical requests:

    | step | PLE in_ids | in_ctx | **PLE out** | RACE | **PLE in_hs** | layer-0 return (row0) | ple_embedding |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | prefill | identical | identical | identical | — | identical | identical | identical |
    | decode 1 | identical | identical | **identical** | none | **DIFFERS** | identical | identical |
    | decode 2, 3 | identical | identical | identical | none | differs | identical | identical |

    - **The PLE is deterministic at every step**, offload lookup included (`out` identical, no race
      flag, `ple_embedding` row0 identical). Given identical inputs it returns identical output.
    - **Its INPUT differs at decode step 1** while layer 0's returned tensor is identical. The code
      between them is `attn_hc.combine(hidden_states, prev_block_output, prev_injection)` —
      `nvidia/model.py:288`, with the comment *"PLE adds directly to the multi-stream state, so
      pending HC state must be materialized before the addition."*
    - **This resolves the weak-fingerprint puzzle**: hyperconnections defer each layer's block
      output as pending state (`prev_block_output`, `prev_injection`) carried *alongside* the
      returned tensor. The layer hook hashed only the return, so layer 0 looked identical all
      night while its pending block output could differ. Layer 0's GDN state is identical
      (finding 24), so the differing pending output must come from **after** the GDN: layer 0's
      MLP — the **MoE** — or the HC combine itself, in the single-token decode shape.
    - `convstate` in this run is a whole-buffer hash across different slots and is not comparable;
      instrument limitation, disregarded.

    Consistent with the standing memory that Flash-Next (MoE) diverges at c=1 while the dense
    27B does not, attributed then to MoE routing ties. Next arm: hash every submodule of layer 0
    at decode step 1 — `mlp.gate` (router), `mlp.experts`, `mlp.shared_expert`, the HC ops.

26. **ROOT CAUSE, component level: `mlp.experts` — the fused MoE expert kernel — is
    nondeterministic at small M.** `LAYER0SUB` arm (cache off, spec off, eager, triton; v3 row0
    hashes on all 18 layer-0 submodules, 106 modules hooked; raw in
    `notes/data/LAYER0SUB-run.txt`), three identical requests:

    | layer-0 submodule (forward order) | prefill (M=55) | decode 1 (M=1) | decode 2 |
    | --- | --- | --- | --- |
    | attn_hyper_connection.* | identical | identical | identical |
    | linear_attn.in_proj_qkvz / in_proj_ba / norm / out_proj / linear_attn | identical | identical | identical |
    | mlp_hyper_connection.* | identical | identical | identical |
    | **mlp.gate** (router logits) | identical | **identical** | identical |
    | mlp.shared_expert (+gate, gate_up, act, down) | identical | identical | identical |
    | **mlp.experts** | identical | **DIFFERS** | DIFFERS |
    | mlp | identical | DIFFERS | DIFFERS |

    - **Routing is excluded**: `mlp.gate` is bit-identical, so the same experts are selected with
      the same weights; the *expert computation/combine* differs. Not routing ties.
    - **Everything upstream of the experts is deterministic at M=1** — GDN, hyperconnections,
      shared expert — so this is the kernel, not its inputs.
    - **Shape-dependent**: identical at M=55, divergent at M=1. The signature of a small-M code
      path with a nondeterministic reduction (atomic-add combine or split-K).

    **What this unifies:**
    - decode always runs M=1 → diverges from step 1 (findings 20, 21, 24, 25)
    - cache-off prefill is one 55-token pass → deterministic, within and across starts
    - `LAUNCHBLOCK` negative: intra-kernel atomics are indifferent to launch order (finding 13)
    - `runtime_determinism.py` clean: it never ran the MoE kernel (finding 19)
    - layer 0's GDN state identical, its return identical, its *deferred HC block output*
      different (finding 25): the MoE output is exactly what the deferred block output carries
    - the fused cuda GDN op diverging at prefill (finding 23) is a *separate* nondeterminism in
      a different kernel — the only result this does not absorb
    - **cache-ON prefill divergence, hypothesis**: `_mamba_block_aligned_split` splits the prefill
      into block-aligned chunks; small chunks put the MoE on its small-M path. Testable: the same
      submodule arm with the cache on (`LAYER0SUB_CON`, launched) — `mlp.experts` first again,
      and a chunked prefill in the log, would close it.

    **The kernel**: the arm's log names it — `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`
    (`nvfp4.py:291`), with `MoEPrepareAndFinalizeNoDPEPModular`. So the nondeterministic
    small-M path is FlashInfer's CUTLASS NVFP4 grouped-GEMM MoE on sm_121.

    **Not yet established**: the
    per-start bias of the acceptance flip (finding 9) — atomics give per-execution noise, not a
    per-start tilt, so something else still contributes there.

27. **CACHE-ON PREFILL DIVERGENCE IS THE SAME KERNEL: the prefill is chunked 52+3, and
    `mlp.experts` is the first differing module in BOTH chunks.** `LAYER0SUB_CON` (prefix cache
    ON, otherwise identical to `LAYER0SUB`; raw in `notes/data/LAYER0SUB_CON-run.txt`):

    - **Shapes prove the chunking**: cache off, each request's prefill is one `(55, 10240)` pass;
      cache on, it is `(52, 10240)` then `(3, 10240)` — a block-aligned split leaving a 3-token
      tail (`_mamba_block_aligned_split`; the log confirms chunked prefill enabled).
    - **Per submodule, full-tensor hash across 3 requests**, both chunks: every hyperconnection
      mix, the entire GDN chain, `mlp.gate`, the shared expert — identical. `mlp.experts` —
      **DIFFERS**. In the 52-token chunk *and* the 3-token tail.

    **Revision to finding 26's mechanism**: it is not simply "small M". `mlp.experts` differs at
    M=52, M=3 and M=1, yet the cache-off single pass at M=55 is bit-identical across six
    independent arms. What the FlashInfer CUTLASS NVFP4 MoE does differently at 52 vs 55 is open
    (kernel-config selection by M? padding rows via `num_tokens_padded`? a workspace not
    re-zeroed between chunks?). **What is established**: `mlp.experts` is the first differing
    module in *every* diverging pass measured — cache-on prefill (both chunks) and decode — and
    is identical in the one non-diverging pass. Its inputs are identical in all of them.

    **Unification complete at the component level**: one kernel explains the cache-on prefill
    divergence, the decode divergence, the deferred-HC puzzle, the LAUNCHBLOCK null and the clean
    runtime test. The mamba/align state machinery is exonerated as a *cause* — its role was
    chunking the prefill so the MoE ran in a diverging configuration.

28. **MoE backend A/B, round 1: `marlin` and `humming` are WORSE than the incumbent.** Same probe
    as finding 21 (cache off, spec off, eager, max_tokens=4, per-token signatures), backend
    verified from each arm's own `NvFp4 MoE backend` log line:

    | backend | prefill (token 1) | decode (token 2+) |
    | --- | --- | --- |
    | `flashinfer_cutlass` (incumbent, 6 arms) | **identical** | differs |
    | `marlin` | **differs** | differs |
    | `humming` | **differs** | differs |

    Marlin's MoE accumulates with atomic adds (the `VLLM_MARLIN_USE_ATOMIC_ADD` lever we enable for
    speed on other models), so prefill nondeterminism is expected. Humming declares
    `_supports_batch_invariance`, but that path needs `VLLM_BATCH_INVARIANT=1`, which this
    architecture cannot run (finding 7); its default path diverges. Neither is a mitigation.
    `cutlass` (`VLLM_CUTLASS`, verified loaded) **crashes at engine init** on sm_121 —
    `Triton Error [CUDA]: an illegal memory access was encountered` — so it is unusable here, not
    merely nondeterministic. `triton` / `triton_unfused` are **rejected** at init — `not supported for NvFP4 MoE`. The
    selectable NvFP4 set is `cutlass`, `flashinfer_trtllm`, `flashinfer_cutlass`, `flashinfer_cutedsl`,
    `flashinfer_b12x`, `marlin`, `humming`, `emulation`; with trtllm (SM121 garbage bug) and
    cutedsl/b12x (vetoed on this checkpoint) excluded, **every serving-grade backend is measured
    and none is deterministic**. `emulation` (dequantised) runs as a control (`moeab3`). `vllm_cutlass` was an invalid CLI
    name (the choice is `cutlass`) and died at argparse — corrected.

29. **Upstream cousin: [flashinfer#3957](https://github.com/flashinfer-ai/flashinfer/issues/3957)**
    — nvfp4 unified-MoE, *silent* out-of-bounds device write from one call that corrupts a later
    one; the victim config has **3 tokens** (top-k 8); suspected root cause *"atomic scatter-add
    finalize in cutlass DSL nvfp4"*; passes in isolation, fails deterministically after ~23 other
    shapes; open, classed a release blocker. Differences from ours: SM100 (B200) not sm_121, and
    the cutlass-DSL / trtllm_fp4_routed variants rather than `FLASHINFER_CUTLASS`. Similarities:
    same kernel family, a 3-token shape, an atomic finalize, silent corruption. It also suggests a
    reading of finding 27's open question — deterministic at the first shape (55, cache off) but
    divergent once shapes vary (52 → 3 → 1) is what cross-call state would look like, not what a
    pure M-threshold would.

30. **CLOSED AT THE KERNEL: `--moe-backend emulation` is fully deterministic — all 4 tokens
    identical across 3 requests** (backend verified `EMULATION` in the log; cache off, spec off,
    eager). The dequantised expert path removes the divergence entirely, prefill and decode. So
    the nondeterminism is in the NVFP4 MoE kernels — every serving-grade one measured — and not
    in the model, the recurrent state, the PLE, the hyperconnections, the scheduler or the
    runtime. **Emulation costs +17% decode** (`EMUCOST`, 8-turn agent loop, c=1, no spec, cache on:
    `flashinfer_cutlass` 43.92 ms/tok — on the 12-arm reference — vs `emulation` 51.38; n=1 for
    emulation, TTFT/concurrency unmeasured; raw in `notes/data/EMUCOST-run.txt`). I had written
    "far too slow to serve with" without measuring — withdrawn. It is a serving-viable
    deterministic mode at modest cost.

    **Final backend table** (same probe; backend verified per arm):

    | `--moe-backend` | prefill | decode |
    | --- | --- | --- |
    | `flashinfer_cutlass` (auto) | identical | differs |
    | `marlin` | differs | differs |
    | `humming` | differs | differs |
    | `cutlass` | crashes at init (illegal memory access) | — |
    | `triton`, `triton_unfused` | rejected for NvFP4 | — |
    | **`emulation`** | **identical** | **identical** |

    **Stop condition (a) met**: root cause located, deterministic configuration demonstrated,
    upstream report drafted (`upstream-report-draft.md`).

31. **FIX CANDIDATE: FlashInfer's own `use_fused_finalize=False`.** `flashinfer/fused_moe/core.py`
    documents the knob on `cutlass_fused_moe`: *"The fused epilogue reduces expert outputs via
    non-associative atomics, so results are not deterministic run-to-run. Set to False to use the
    non-fused, deterministic finalize path."* Default `True`; **vLLM never passes it**
    (`experts/flashinfer_cutlass_moe.py:367`, the call has no such kwarg; the lazy wrapper forwards
    `**kwargs`). So every arm in this investigation ran the atomic finalize by default — and the
    kernel's own authors name our mechanism. `tools/determinism/fusedfinalize_patch.py` adds the
    kwarg, env-gated on `VLLM_MOE_DET_FINALIZE=1`. Validation running (`detfin`): cache off /
    cache on / cache on + MTP n=5, per-token probe; prediction: all tokens identical in all three.
    If it holds, the cost to measure is finalize-only, which should be far below emulation's +17%.

32. **First fix attempt died in FlashInfer, not in the model.** With `use_fused_finalize=False`
    alone, engine init failed: `Check failed: … Invalid gemm2 profile id: 50`. The in-process
    autotuner enumerates tactics from the runner's `get_gemm1/2_tactic_count()` and then the C++
    runner rejects the chosen GEMM2 id — Python's tactic table and the non-fused runner's disagree.
    (The shipped `tuning_configs` are B200/GB200-only, so this is not a stale on-disk cache.)
    FlashInfer's own bypass: `profile_ids=[-1, -1]` — "keeps the default tactic" — and the code
    special-cases `-1` throughout. **Patch v2** passes both, env-gated; validated install/remove on
    a copy; single arm `DETFIN2` running (cache off, spec off, eager). If it *starts*, the
    non-fused path works on this build with default tactics; if all 4 tokens are then identical,
    the fix is confirmed and the remaining questions are its cost and whether tuned tactics can be
    restored for it. Reportable upstream on its own: `use_fused_finalize=False` cannot be used
    with the autotuner on sm_121 in this FlashInfer.

33. **The non-fused finalize path cannot start on this FlashInfer build — three attempts.**
    (1) `use_fused_finalize=False` → `Invalid gemm2 profile id: 50` at init; (2) plus
    `profile_ids=[-1,-1]` → same, id 50; (3) plus the autotune sweep skipped for
    `trtllm::fused_moe::gemm1/gemm2` (`VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS`) → still dies, **id 48**.
    With no tuner and no caller-chosen tactic, the invalid id comes from FlashInfer's own
    default-tactic resolution, which evidently indexes the *fused* runner's GEMM2 table while the
    non-fused runner checks against its own shorter one. This is a second, independent FlashInfer
    defect on sm_121 NVFP4: **the documented deterministic finalize is unusable**. The deterministic
    serving option on this box therefore remains `--moe-backend emulation` (+17% decode). Both
    defects belong in the upstream report: the atomic finalize's nondeterminism (documented, but
    the default and the only working path) and the broken opt-out.

    **Mechanism, from the shipped JIT source**
    (`flashinfer/data/csrc/fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu:866-869`):
    the GEMM2 check is skipped when `id2 == -1`. Our `profile_ids=[-1,-1]` therefore never reached
    it — FlashInfer's Python side substitutes a concrete id derived from
    `get_gemm2_tactic_count()` (core.py:414), which evidently reports the *fused* runner's table
    size while the C++ runner built with `use_fused_finalize=False` checks against its own,
    shorter `mGemm2TacticCount`. **A local fix is feasible** (source is shipped, JIT rebuilds) but it
    is a FlashInfer binding fix — make the tactic-count getters honour `mUseFusedFinalize`, or
    pass the caller's `-1` through untouched — plus a JIT-cache rebuild on sm_121 (see
    [[flashinfer-jit-oom-after-driver-upgrade]] for the OOM guard). Not attempted tonight.

34. **The non-fused runner is fine; a persisted autotune cache was feeding it fused-runner ids.**
    Direct probe of the sm_121 JIT module (`tools/determinism/tactic_probe.py`, run as `llm`,
    dtype trio bf16 / int64 / bf16 per `isNvfp4Quant()`):

    | `use_fused_finalize` | GEMM1 | GEMM2 | valid GEMM2 ids |
    | --- | ---: | ---: | --- |
    | True | 20 | 40 | 20-59 |
    | False | 20 | 20 | 20-39 |

    The failing ids 48 and 50 lie only in the *fused* range. With tuning skipped they could not
    come from a sweep, and the C++ skips the check for `-1`, so they came from a **cached** tactic:
    vLLM persists FlashInfer autotune results (`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`), earlier fused
    arms wrote entries for these shapes, and the cache key omits `use_fused_finalize`. Finding 33's
    "getter" hypothesis is withdrawn; the defect is the cache key. **Fix attempt 4** (`DETFIN4`):
    non-fused finalize with a fresh, empty cache dir and tuning enabled. Reportable: the autotune
    cache key must include `use_fused_finalize` (or the runner's tactic-table identity).

35. **Upstream check (in parallel with attempt 4): no existing fix for either defect, but the
    cache-key defect is documented almost verbatim.** vLLM `main` still does not pass
    `use_fused_finalize` (call kwargs verified on the live file). FlashInfer
    [#3367](https://github.com/flashinfer-ai/flashinfer/pull/3367) (0.6.13): *"The persistent
    autotune file cache key was constructed as a 3-tuple (custom_op, runner_class, profile),
    intentionally dropping hash(runner) for cross-process stability, but unintentionally also
    dropping extras"* — it added `get_cache_key_extras()` for **`TrtllmGemmRunner` only**
    (`use_8x4_sf_layout`). The cutlass `MoERunner` never got one, so fused and non-fused runners
    share file-cache entries: exactly our 48/50. The real fix is therefore a few lines of
    FlashInfer Python — `get_cache_key_extras()` on `MoERunner` returning `use_fused_finalize`
    (and the other tactic-table-changing flags) — the same shape as #3367, no C++.
    `tools/determinism/moe_cachekey_patch.py` implements it (not env-gated: widening a key is
    pure correctness). Other related, not fixes: flashinfer#3957 (atomic finalize, 3-token
    victim), #4043 (autotuner hash collisions), #2501 (autotune fails for W4A8 cutlass MoE),
    #3537 (tuner picks slower tactics), #3935 (SM120 regression suspected on #3367).

36. **FIX CONFIRMED: `use_fused_finalize=False` is deterministic on the production kernel.**
    `DETFIN4` (FlashInfer CUTLASS NVFP4 MoE, verified in the log; cache off, spec off, eager; fresh
    autotune cache dir): **all 4 tokens identical across 3 requests** — the same result as
    `emulation`, on the fast path. Attempts 1-3 failed only because the shared persisted autotune
    cache handed the non-fused runner fused-range tactic ids (finding 34/35). Next: the proper
    cache-key fix (`moe_cachekey_patch.py`) validated against the *shared default cache*, in all
    three shapes (cache off / cache on / cache on + MTP n=5), plus the cost on the 8-turn agent
    loop against 43.92 (fused) and 51.38 (emulation) ms/tok.

37. **COMPLETE FIX VALIDATED in all three shapes, against the SHARED default autotune cache.**
    Both patches applied — vLLM `use_fused_finalize=False` (env-gated) + FlashInfer
    `MoERunner.get_cache_key_extras()` — no cache-dir override, backend `FLASHINFER_CUTLASS`
    verified per arm:

    | arm | shape | result |
    | --- | --- | --- |
    | `DETFIN5` | cache off, spec off, eager | **all 4 tokens identical x3** |
    | `DETFIN5_CON` | prefix cache ON (the 52+3 chunked prefill), graphs on | **all 4 tokens identical x3** |
    | `DETFIN5_MTP` | prefix cache ON + MTP n=5 (production shape) | **all 4 tokens identical x3** |

    The cache-key patch does against the shared cache what a fresh directory did in `DETFIN4`,
    so attempts 1-3's failures are fully explained and closed. Every configuration that diverged
    in this investigation is now bit-reproducible on the fast kernel. Cost measurement
    (`DETCOST`, 8-turn agent loop) running.

38. **The fix costs +3.6% decode.** `DETCOST` (8-turn agent loop, 130 tok/turn, c=1, no spec,
    cache on, both patches, backend verified): **45.50 ms/tok** vs 43.92 fused (12-arm reference
    42.8-47.7) vs 51.38 emulation. n=1, but no-spec arms reproduce to ~1.10x, and 45.50 sits
    inside the fused reference band's upper half. Only the finalize changes — the expert GEMMs stay
    tuned — which is why it lands far below emulation's +17%. TTFT and concurrency unmeasured.

39. **The cache-key defect is already fixed upstream — our FlashInfer patch is a backport.**
    FlashInfer `main` has `MoERunner.get_cache_key_extras()` with the comment *"Include those
    options here to prevent runners with identical tensor profiles from reusing incompatible saved
    tactics"*, returning dtypes, top-k, parallel ranks, quantization flags, `min_latency_mode`, …
    and `use_fused_finalize`. Pinned by file content per ref: absent at #3984 (2026-08-06),
    present at #4106 (2026-08-13); **first release v0.6.18rc2, in v0.6.18 final; absent in our
    0.6.17**. Consequences: (a) no FlashInfer issue needed — cite the fix instead; (b) upgrading
    FlashInfer to 0.6.18 would bring it, but 0.6.18 drops the SM121a cubins from the aarch64 JIT
    cache on this box ([[flashinfer-jit-oom-after-driver-upgrade]] / working-config memory), so
    the backport stays until that is resolved; (c) the **vLLM** side is the only new upstream item:
    `main` still never passes `use_fused_finalize`. No rebase of the box is needed for the fix.

40. **The fix does NOT stabilise MTP — the per-start bimodality is a separate defect.** `MTPFIX`
    (both patches installed, `VLLM_MOE_DET_FINALIZE=1`, MTP n=5, 8-turn agent loop, prefix cache
    on, backend `FLASHINFER_CUTLASS` verified in all three logs), three server starts:

    | start | ms/tok | acceptance | mean accept len |
    | --- | --- | --- | --- |
    | a | 67.25 | 9.3 % | 1.47 |
    | b | 32.45 | 66.3 % | 4.32 |
    | c | 49.78 | 25.7 % | 2.28 |

    Spread 2.07× across starts — the same spread as without the fix (1.83×, `AC1..5`). With a
    bit-deterministic target the drafter still alternates between the good and the bad regime per
    start, and per turn inside a start (arm a: turns 1–6 at ~10 s, turn 7 at 4.0 s when the
    prefix-cache hit count jumped 4848→6464). So the explanation in the draft report — "acceptance
    flips because the target moves under the drafter" — is **withdrawn**: the MoE nondeterminism
    is real and fixed, but it is not what makes MTP unstable. Whatever sets the regime is chosen
    per start and per turn independently of the target's arithmetic; the prefix-cache-hit
    coincidence points at drafter state under the block-aligned mamba split (the #47861 thread),
    not at the MoE. Raw: `notes/data/mtpfix.txt`.

41. **The MTP flip is per REQUEST, and none of async scheduling / CUDA graphs / ring widening /
    prefix cache is the switch.** `MTPROOT` (12 starts, MTP agent loop, per-turn acceptance from
    the metrics deltas, every arm's config line verified in its log):

| arm | ms/tok | acceptance | accept len | per-turn pattern (F ≥40 %, s <40 %) | healthy turns |
| --- | --- | --- | --- | --- | --- |
| NOASYNC_a | 56.55 | 17.5 % | 1.87 | `FssssFss` | 2/8 |
| NOASYNC_b | 52.60 | 19.8 % | 1.99 | `ssFFssFs` | 3/8 |
| NOASYNC_c | 56.01 | 17.2 % | 1.86 | `sssFsFsF` | 3/8 |
| EAGER_a | 61.62 | 13.8 % | 1.69 | `ssssFFss` | 2/8 |
| EAGER_b | 50.32 | 24.5 % | 2.22 | `ssFsFsFF` | 4/8 |
| EAGER_c | 33.15 | 65.1 % | 4.26 | `FFFFFFFF` | 8/8 |
| N4_a | 55.42 | 15.9 % | 1.64 | `ssssssFF` | 2/8 |
| N4_b | 52.51 | 19.9 % | 1.80 | `ssFFssss` | 2/8 |
| N4_c | 57.05 | 16.6 % | 1.66 | `ssssFsFs` | 2/8 |
| NOCACHE_a | 68.08 | 18.3 % | 1.92 | `sssFsFFs` | 3/8 |
| NOCACHE_b | 49.07 | 59.8 % | 3.99 | `FFFFFFFF` | 8/8 |
| NOCACHE_c | 69.92 | 17.4 % | 1.87 | `ssssFsFF` | 3/8 |

healthy turns: n=42, acceptance 40–88 %; broken turns: n=54, 3–25 %

    Every turn is in one of two clean states — healthy (≈50–77 % accepted, accept len ≈3.5–4.9)
    or broken (≈3–21 %, len ≈1.1–1.9) — with almost nothing between. The state holds for the
    request's lifetime and is drawn afresh per request; three starts (EAGER_c and NOCACHE_b here,
    MTPFIX_b before) were healthy on all 8 turns, so something at start decides whether the per-request
    draw can come up broken at all. The earlier "per-start bias" (finding 40) was 8-turn
    sampling of this per-request draw. NOCACHE (prefix caching off, 0 hits, every turn a full
    re-prefill) flips just the same, which removes the stale-GDN-state-on-resume lead
    (#53142/#54076/#53798) as the cause of *this* symptom; those PRs remain a real defect on this
    build (`cache_config.block_size` = the QSA ring's 16, confirmed in `core.py`) and are queued
    for their own validation. Raw: `notes/data/mtproot.txt`. Survivors:
    the drafter's unzeroed per-request QSA ring block (claimed FIFO, excluded from zeroing,
    polluted by warmup — `mtpring` queued), a shape-bucketed drafter kernel tactic per start,
    and the drafter's own top-k.

42. **The per-request draw does not follow the prompt.** `BASE_replay` (one start, unpatched):
    pass 1 runs the 8-turn loop live; passes 2 and 3 resend the byte-identical conversation
    (pass-1 outputs substituted). Patterns: `ssssssss`, `FssssFFF`, `ssFssFFs`. Same bytes in,
    different healthy/broken draw out — so the state is not a function of prompt content or
    length (which also removes a shape-bucketed kernel tactic as the per-request switch). What
    is left is what the request is *given*: its KV/ring blocks and its slot, or timing. Raw in
    `notes/data/mtpfix2.txt` (copied when the run ends).

43. **The align PRs (#54076 + #53798) are the first thing that moves the rate — signal, not yet
    a result.** `MTPFIX2`, both patches installed (adapted to this build: `MambaSpec` import, no
    internal-checkpoint mode), MTP n=5, prefix cache on:

    | arm | healthy turns | pattern |
    | --- | --- | --- |
    | ALIGNFIX_a | 2/8 | `sFsFssss` |
    | ALIGNFIX_b | 8/8 | `FFFFFFFF` |
    | ALIGNFIX_c | 7/8 | `FFFFFsFF` |
    | ALIGNFIX_replay pass 1/2/3 (one start, identical prompts) | 8/8, 7/8, 7/8 | `FFFFFFFF` `FsFFFFFF` `sFFFFFFF` |

    39 of 48 turns healthy (81 %) against 42 of 96 (44 %) unpatched, over 4 starts. Not an
    elimination (c, and two replay passes, each had a broken turn), and 4 starts of a variable
    that came up 8/8 healthy in 3 of 15 unpatched starts is not proof of a rate change — the
    user's standard: no call from three runs. Two things it cannot be, if it holds: the resume
    path alone (NOCACHE removed it and stayed at 14/24), so the effect would sit in what else the
    patch changes — prefill chunk ends at every 1616 boundary (chunks ≤1616 instead of ≤4096) and
    the state seed for resumed requests. `MTPFIX3` (3 more starts of both, then seed-only ×2,
    split-only ×2) is queued to confirm and separate. Raw: `notes/data/mtpfix2.txt`.

44. **Under MTP the generated text is not reproducible even with the deterministic MoE.** `MTPQ`
    (det finalize installed, MTP n=5, 2 starts × 3 passes of byte-identical prompts): pass 2 vs
    pass 3 — same cache state — differ in the generated text on 8 of 8 turns, in start a and in
    start b, in healthy and in broken turns alike. So a text hash cannot tell "target corrupted"
    from "drafter broken": MTP's verify shapes depend on how many drafts were accepted, and the
    kernels are deterministic but not batch-invariant, so any acceptance difference changes
    downstream logits and the text forks. The target-or-drafter question needs a no-spec
    reference and the *position* of first divergence per turn (`MTPQ2`, queued). Patterns:
    a `ssFFsFss` `sssFssss` `ssFsFsss`, b `ssFssFss` `ssssFssF` `sssFssss` — 6 of 48 healthy,
    the lowest rate of any start so far. Raw: `notes/data/mtpq.txt`.

45. **The broken state does not corrupt the target's text — the defect is in the drafter.** `MTPQ2`:
    a no-spec reference conversation with the deterministic MoE, then MTP n=5 (same deterministic
    MoE) replaying the reference prompts, 2 starts × 3 passes, first divergent character vs the
    reference per turn (ref lengths 1351/1154/1428/610/1351/1233/1365/584):

    | | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | a p1 `sFsssFss` | 56 | 194 | 142 | 15 | = | 44 | 15 | 15 |
    | a p2 `sssssFFF` | 56 | 93 | 142 | 15 | 84 | 15 | 44 | 15 |
    | a p3 `ssssssss` | 56 | 986 | 142 | 15 | 84 | 33 | 104 | 15 |
    | b p1 `Fsssssss` | 127 | 878 | 142 | 15 | 1279 | 33 | 104 | 15 |
    | b p2 `sssssssF` | 56 | 150 | 142 | 77 | = | 140 | 44 | 15 |
    | b p3 `FssssFFs` | 56 | 34 | 142 | 15 | 79 | 140 | 44 | 15 |

    The fork position is a property of the turn (56, 142, 15, 15 recur in every pass whatever the
    state), i.e. near-ties in the reference where MTP's verify shapes tip the argmax; a broken turn
    reproduced the reference exactly twice (a/p1 t5, b/p2 t5) and stayed with it to char 1279 once.
    Healthy and broken turns fork at the same places. So the broken state changes how many drafts
    the target accepts, not what the target says: the drafter is wrong, the target is not. Also:
    even with a bit-deterministic MoE, MTP output is not greedy-equivalent on this stack (cf. the
    new upstream CI test #54893, which asserts that equality on its hardware). Raw:
    `notes/data/mtpq2.txt`, reference texts `notes/data/mtpq2-ref.json`.

46. **Bug A confirmed and split: EITHER align patch alone gives ~90 % healthy — the seed (#53798)
    and the chunk split (#54076) act on the same wrong-block-size defect.** `MTPFIX3`, MTP n=5, prefix cache on:

    | arm | pattern | healthy |
    | --- | --- | --- |
| ALIGNFIX_d | `FFFFFFFF` | 8/8 |
| ALIGNFIX_e | `FFFFFFFF` | 8/8 |
| ALIGNFIX_f | `FFFFFFFF` | 8/8 |
| SEEDONLY_a | `FFFFFFFF` | 8/8 |
| SEEDONLY_b | `FFFFFFss` | 6/8 |
| SPLITONLY_a | `FsFFFFFF` | 7/8 |
| SPLITONLY_b | `FFFFFFFF` | 8/8 |

    Totals: both patches, 7 starts + 3 replay passes, **63/72 healthy (88 %)**; seed-only
    14/16; split-only 15/16; unpatched 42/96 (44 %), and 3 of 15 unpatched starts were 8/8
    against 5 of 7 patched. Past the three-run bar for a rate change. Mechanism, from the code:
    on this build `cache_config.block_size` is the QSA ring's capacity (16), so a resumed request's
    running mamba state index was seeded as `(num_computed − 1) // 16` instead of `// 1616` —
    pointing the align precopy at the wrong block-table column, i.e. another request's (or a
    stale) GDN state. With MTP every agent turn is a resume, and the drafter's inputs come from a
    target running on a wrong state until the next checkpoint realigns it — the per-turn draw.
    Not an elimination: 9 broken turns remain under both patches (bug B, per request, also present
    with the cache off where the seed path is never taken). Raw: `notes/data/mtpfix3.txt`.

47. **The ring block is not the switch.** `MTPRING` (unpatched align path; ring block id logged per
    request; then ring blocks included in allocation zeroing via `VLLM_RING_ZERO=1`):

    ```
RINGID_a: claims=8 turns=8  blk4:FFF  blk28:FFF  blk39:F  blk46:F
RINGID_b: claims=8 turns=8  blk4:ssF  blk26:ss  blk28:ss  blk44:F
RINGZERO_a: claims=8 turns=8  blk4:FFF  blk24:FF  blk28:Fs  blk44:F
RINGZERO_b: claims=8 turns=8  blk4:FFF  blk24:FF  blk28:FF  blk44:F
RINGZERO_c: claims=8 turns=8  blk4:sss  blk28:ss  blk43:Fs  blk49:s
    ```

    Same block id draws both states (RINGID_b: block 4 → s, s, F; block 28 → s, s). With zeroing,
    starts were 7/8, 8/8 and 1/8 (c: 67 ms/tok) — the two good starts are within the unpatched
    luck (3 of 15 all-healthy) and the 1/8 start with zeroed rings is an existence proof that
    zeroing does not remove the draw. Ring inheritance is out; the code-trace's rank-1
    candidate for bug B does not hold. Raw: `notes/data/mtpring.txt`.

48. **Not a timing race.** `MTPRACE`, `CUDA_LAUNCH_BLOCKING=1` verified on the units (every kernel
    launch serialised), MTP n=5, unpatched align path:

    | arm | pattern | healthy |
    | --- | --- | --- |
| RACE_a | `FFFFFFFF` | 8/8 |
| RACE_b | `ssssssFs` | 1/8 |
| RACEEAGER_a | `FFFFFFFF` | 8/8 |
| RACEEAGER_b | `ssFsssss` | 1/8 |

    The draw survives full serialisation, with and without CUDA graphs. Bug B is not an ordering
    problem between kernels. Worth noting (n=4): under launch blocking every start was all-or-
    nothing (8/8, 1/8, 8/8, 1/8) — the per-turn flipping seen elsewhere did not appear. Raw: `notes/data/mtprace.txt`. Remaining for bug B: the
    drafter's own computation (top-k tie behaviour, input-slot binding) — `MTPDH` bisects inside
    the drafter next.

49. **Bug A seen from inside: after a prefix-cache resume the target's hidden state is not
    reproducible, even with the deterministic MoE.** `MTPDH` / `DH_a` (unpatched align path, det
    finalize, `--enforce-eager`, MTP n=5, identical prompts × 3 passes; `tools/determinism/
    drafthash_patch.py` hashes the drafter's input hidden state and 31 submodule outputs on each
    turn's draft-prefill call and the six draft calls after it). Pass 2 vs pass 3 (same cache
    state, same tokens): the drafter's **INPUT** — the target's multi-stream hidden state for the
    first token of the resumed chunk — already differs on all 8 turns, so every module after it
    differs too and nothing inside the drafter can be bisected on this build. The text survives
    (45) but the numerics do not; that is the align seed/split defect (46) delivering a different
    GDN state on every resume. Bisection of bug B needs a reproducible input: `MTPDH2` runs the
    same instrument with the align patches installed (DH2) and with the cache off (DH3). Raw:
    `notes/data/fnext-DH_a.log.txt`; report: `tools/determinism/drafthash_report.py`.

50. **The nondeterminism enters at the prefill chunk boundary, not at the resume.** `MTPDH2`:
    - `DH2_a` (align patches ON, cache on): eight prefill chunks starting at the cached position
      4848, same token — eight distinct row-0 hidden states. The resumed state is different on
      every request even with the align fix.
    - `DH3_a` (cache OFF, no resume at all): chunk 1 (4096 tokens at position 0) has the **same**
      row-0 hidden hash on every request of every pass (`8cb2cf5b4a15`); chunk 2 (same tokens, at
      position 4096) has a **different** hash every time.
    So with the deterministic MoE the first chunk of a prefill is bit-reproducible and the second
    is not: whatever carries state across the chunk boundary (GDN/conv/PLE running state written
    at the end of chunk 1, read at the start of chunk 2) delivers a different state each time. That
    is consistent with every fact about bug B: per request (every prefill draws anew), prompt-
    independent, not a config knob, not a race between launches, not the ring, text mostly
    robust (45) but the drafter's input differs (49). Bug A (46) is the same class at the cache
    boundary — the seed/split fix made it *rarer*, not exact. `MTPDH3` next: one-chunk prefill
    (`--max-num-batched-tokens 8192`) vs four chunks (2048), hashing also the last row of each
    chunk (kernel-internal vs handoff). Raw: `notes/data/fnext-DH2_a.log.txt`, `fnext-DH3_a.log.txt` (when done).

51. **Not the chunk handoff either — the long prefill itself is nondeterministic.** `MTPDH3`
    (cache off, det MoE, eager, hashes of the FIRST and LAST row of every draft-prefill chunk):
    - `B8192_a` (`--max-num-batched-tokens 8192`, the 7503-token turn-1 prompt is ONE chunk, no
      handoff): row 0 identical across passes, the **last row differs** between identical
      7503-token requests. Patterns `ssssFsFF` `FssFsFFF` `ssssFsFs` — no mitigation.
    - `B2048_a` (four chunks): first chunk's row 0 identical, its last row and every later chunk
      differ. Patterns `ssssssFs` `Fsssssss` `FssFsFss`.
    So a kernel in the prefill produces different values at late positions for identical inputs
    while its first rows are exact — position-dependent nondeterminism inside one forward, with
    the MoE finalize already deterministic. All earlier "forward pass is deterministic" results
    were on a 55-token prompt (the audit's rank-1 caveat, now confirmed as the blind spot).
    `MTPLH` bisects it: every layer and submodule of the target hashed over the full tensor on
    three identical 7.5k-token requests. Raw: `notes/data/mtpdh3.txt`, `fnext-B8192_a.log.txt`,
    `fnext-B2048_a.log.txt`.

52. **The long-prefill nondeterminism is in the QSA attention layers, not the GDN.** `MTPLH` /
    `LHLONG`: no spec, cache off, eager, one 8192-token chunk, det MoE, the 7503-token turn-1
    prompt three times, full-tensor hash of every decoder layer's output:

    | module | 3 requests |
    | --- | --- |
    | layers.0, 1, 2 (linear_attention ×3, PLE at layer 2) | **identical**, all 7503 rows |
    | layers.3 (first `full_attention` = QSA) | row 0 identical, full tensor **differs** |
    | layers.4 … 47 | differ (downstream) |

    Top-1 logprob of the next token: −0.0904 / −0.0435 / −0.0449 — the output already moves.
    So over 7.5k tokens the GDN chunked kernels, the PLE and the (now deterministic) MoE are
    exact, and the first QSA layer is not, with a position-dependent signature (row 0 exact).
    That is the sparse-attention path: the indexer's `persistent_topk` (vllm#54521: tie set /
    order varies on sm_121, cooperative top-k never selected on capability 12.x) feeding a
    gather whose summation order then differs, or the split-K attention merge. `LHSUB3` hooks
    every submodule of layers 3 and 7 next. Raw: `notes/data/fnext-LHLONG.log.txt`, `mtplh.txt`.

53. **Bug B located: the QSA indexer (`persistent_topk` block selection) is the first module
    that differs, with bit-identical inputs.** `MTPLH2` / `LHSUB3` (same probe as 52, every
    submodule of layers 3 and 7 hooked, full-tensor hashes, 3 identical 7503-token requests),
    layer 3 in forward order:

    | module | 3 requests |
    | --- | --- |
    | `attn_hyper_connection.*` | identical |
    | `self_attn.qkv_proj` | identical |
    | `self_attn.indexer.index_qk_proj` | identical |
    | **`self_attn.indexer`** (block selection = `qsa_select_paged_tokens` → `persistent_topk`) | **differs** |
    | `self_attn` / `o_proj`, `mlp_hyper_connection`, `mlp.*`, layer output | differ (downstream) |

    Same inputs, different selected blocks; the sparse attention then sums a different key set
    (or the same set in a different order) and the hidden state forks. On sm_121 the cooperative
    top-k is never selected (`is_device_capability_family(120)`), so every selection goes through
    `persistent_topk` — vllm#54521 / #51782, previously seen only as text nondeterminism. Chain
    closed: nondeterministic selection at long context → target hidden state differs per request
    → the drafter (whose only attention is this same QSA path) draws healthy or broken per
    request → MTP acceptance flips. `MTPQSA` tests the two fixes at the call site: canonical
    ordering of the selected blocks (padding kept trailing) and an exact `torch.topk`. Raw:
    `notes/data/fnext-LHSUB3.log.txt`, `mtplh2.txt`.

54. **CONFIRMED: an exact top-k at the QSA selection site makes the entire 7.5k-token forward
    bit-identical.** `MTPQSA`, same probe as 52/53 (3 identical 7503-token requests, det MoE,
    every layer + every submodule of layers 3 and 7 hashed), `tools/determinism/qsafix_patch.py`:

    | selection | next-token logprob ×3 | first differing module |
    | --- | --- | --- |
    | `persistent_topk` (stock) | −0.0904 / −0.0435 / −0.0449 | layers.3.self_attn.indexer |
    | `persistent_topk` + canonical order (`VLLM_QSA_SORT`) | −0.0622 / −0.0610 / −0.0622 | layers.11 (req 2 only); layers 3 and 7 identical incl. indexer |
    | exact `torch.topk` over the visible logits (`VLLM_QSA_EXACT_TOPK`) | −0.0592 ×3 | **none — all 48 layers, all 160 hooked modules identical** |

    So on sm_121 `persistent_topk` returns the selected blocks in a varying ORDER (fixed by the
    sort: layers 3 and 7 become exact) AND, less often, a varying SET (the residual at layer 11
    in one of three requests; fixed by the exact selection). Correction to finding 5: "QSA top-k
    excluded" was measured on a 55-token prompt with the MoE still nondeterministic, so it only
    showed top-k was not the sole source then. Raw: `notes/data/lhsort_hashes.txt`,
    `lhexact_hashes.txt`. The MTP replay arms under each fix follow (`MTP_EXACT`, `MTP_SORT`).

55. **With exact top-k the MTP loop is bit-deterministic; the "flip" becomes a fixed property of
    the turn.** `MTP_EXACT` (exact `torch.topk` at the QSA selection, det MoE, cache off, MTP n=5,
    identical prompts × 3 passes): per-turn acceptance **identical to the decimal in all three
    passes** — 14.4 / 5.3 / 63.2 / 14.4 / 17.4 / 61.2 / 66.0 / 4.8 % (`ssFssFFs` ×3). Compare
    finding 42 (stock: `ssssssss` / `FssssFFF` / `ssFssFFs` for the same prompts). So bug B
    (nondeterministic selection, 53/54) is closed with a validated fix, and what is left is
    **not a draw**: for this conversation turns 1, 2, 4, 5 and 8 have ~5–17 % acceptance every
    time and turns 3, 6, 7 have ~61–66 %. A systematic, reproducible drafter/target disagreement
    that depends on the turn (its prompt, length or position) — a third defect, deterministic
    and therefore bisectable per turn. `MTPDH4` (drafter hashes under exact top-k) gives the
    per-turn prompt lengths and chunking to correlate against. Raw: `notes/data/mtpqsa.txt`.

56. **WITHDRAWN.** `DH4_a` ran WITHOUT exact top-k: its log has no `QSAFIX active` line. Two env
    vars had been passed in ONE systemd `Environment=` entry (`"VLLM_DRAFT_HASH=1 VLLM_QSA_EXACT_TOPK=1"`),
    so the first got the value `1 VLLM_QSA_EXACT_TOPK=1` and the second was never set. The eager
    nondeterminism it reported is bug B, nothing new. Rule: one variable per `Environment=` entry,
    and every arm must show its own activation line. Original text kept below for the record.
    ~~Eager mode has one more nondeterministic kernel that compiled mode does not use.~~ `MTPDH4`
    (exact top-k, det MoE, cache off, `--enforce-eager`, drafter hashes, 3 passes): the draft
    PREFILL call is bit-identical across passes (input and every hooked module), but the first
    draft step after it already differs, and the per-turn acceptance differs per pass
    (`sssFssss` / `sFssssss` / `sssssssF`) — whereas the compiled `MTP_EXACT` run (55) was
    identical to the decimal. The eager-only source sits in the drafter's own step after the
    prefill (its unquantised BF16 MoE / eager attention path is the candidate); production runs
    compiled (PIECEWISE), so this does not bear on prod, but the eager hash instrument cannot be
    used to bisect defect C. Raw: `notes/data/mtpdh4.txt`, `fnext-DH4_a.log.txt`.

57. **The full fix stack is clean in the production shape: 24/24 healthy turns, identical to the
    decimal across three passes.** `MTPPAD` / `PAD_PROD`: prefix cache ON + align patches (#54076,
    #53798) + exact top-k at the QSA selection + deterministic MoE finalize (+ the draft-pad fix,
    which 57a shows is inert), MTP n=5, compiled (PIECEWISE), identical prompts × 3 passes:

    ```
PAD_PROD p1: 63.2% 68.0% 60.6% 55.9% 56.5% 58.2% 69.0% 72.4% 
PAD_PROD p2: 63.2% 68.0% 60.6% 55.9% 56.5% 58.2% 69.0% 72.4% 
PAD_PROD p3: 63.2% 68.0% 60.6% 55.9% 56.5% 58.2% 69.0% 72.4%
    ```

    31–35 ms/tok. Against the same prompts unpatched: 42/96 healthy over 12 starts, 32–70 ms/tok.
    57a. The draft-pad candidate (stale trailing slots, `tools/determinism/draftpad_patch.py`) is
    inert: `PAD_EXACT` reproduces `MTP_EXACT` to the decimal on every turn. Defect C — the
    reproducible ~5–17 % turns of finding 55 — occurs only with the prefix cache OFF, i.e. when
    every turn rebuilds the drafter's cache through a full draft prefill; with the cache on, the
    drafter's state is built incrementally during decode and every turn is healthy. So C lives in
    the draft-prefill path (position/content dependent, deterministic), and does not affect the
    served configuration. `MTPACC` (per-step acceptance, cache off) characterises it; the depth
    grid (`MTPGRID2`) runs on the validated stack. Raw: `notes/data/mtppad.txt`.

58. **The depth grid on the full fix stack is bit-reproducible across starts, and four depths are
    broken deterministically.** `MTPGRID2` (prefix cache on, align + exact top-k + det MoE, compiled,
    8-turn loop), three starts — **start c reproduced a and b to the counter at every depth**
    (table shows a/b; `QSAFIX active` verified in all 27 arms):

    | n | ms/tok a / b | acceptance | accept len | pattern a / b | a vs b |
    | --- | --- | --- | --- | --- | --- |
| 0 | 45.4 / 46.2 | - % | - | `-` / `-` | n/a |
| 1 | 42.1 / 39.9 | 85.2 % | 1.85 | `FFFFFFFF` / `FFFFFFFF` | identical counters |
| 2 | 52.1 / 51.1 | 28.2 % | 1.56 | `ssFssFss` / `ssFssFss` | identical counters |
| 3 | 53.7 / 53.5 | 21.6 % | 1.65 | `ssFssFss` / `ssFssFss` | identical counters |
| 4 | 34.0 / 34.4 | 70.4 % | 3.82 | `FFFFFFFF` / `FFFFFFFF` | identical counters |
| 5 | 34.5 / 34.9 | 62.6 % | 4.13 | `FFFFFFFF` / `FFFFFFFF` | identical counters |
| 6 | 72.1 / 72.1 | 8.8 % | 1.53 | `sssFssss` / `sssFssss` | identical counters |
| 7 | 37.8 / 36.8 | 54.2 % | 4.79 | `FFFFFFFF` / `FFFFFFFF` | identical counters |
| 8 | 75.7 / 76.4 | 8.3 % | 1.67 | `ssssssss` / `ssssssss` | identical counters |

    Every depth reproduces its draft/accept counters exactly from start to start, so on this stack
    the MTP loop is deterministic end to end — and n = 2, 3, 6, 8 are broken *by construction*, not
    by luck: same per-turn pattern (`ssFssFss` at 2 and 3 — the defect-C pattern of finding 55,
    turns 3 and 6 healthy) every time. n = 1, 4, 5, 7 are healthy. Healthy/broken does not follow
    ring capacity (8 for n≤4, 16 above), span parity, or tokens-per-step (1+n) in any way I can
    see. A shared torch.compile cache cannot explain a *deterministic* depth function (both starts
    used it), but the fresh-cache arm still runs as the control. Next arms, in order: per-step
    acceptance (where in a turn does n=2 fail?), fresh compile cache, DENSE drafter (llama.cpp's
    MTP head attends densely; ours re-selects through QSA — `densedraft_patch.py`), index sharing
    (SGLang's strategy; `index_share_for_mtp_iteration`). Raw: `notes/data/mtpgrid2.txt`.

59. **The per-turn "healthy/broken" split is a BENCHMARK ARTIFACT of `ignore_eos`.** `MTPACC2`
    (per-step acceptance from the scheduler + saved texts, full fix stack, n=2 and n=5, cache on):
    - At n=2 a "broken" turn accepts normally for the first ~12 steps (`2102022220122…`), then runs
      **50–90 consecutive steps with 0 accepted**, sometimes recovering to full 2s at the end.
    - The saved text shows why: the model's real answer is **30–40 tokens** ("The code lacks X, so
      …"). The loop sends `ignore_eos: true, max_tokens: 130`, so the target keeps generating past
      its end-of-turn token. What it produces after EOS is one of two near-tie continuations:
      the chat-template **restart** (`<|im_start|>user … <|im_start|>assistant <think> …` followed by
      a verbatim repeat of the answer — trivially predictable, drafts accepted at ~100 %) or a
      **wall of `<|im_start|>`** tokens (never predicted by the drafter, 0 accepted).
    - Answer fraction of each turn's text: 7–30 %. Filler type per turn matches the state exactly:
      ACC2on turns 2/5/7/8 = spam = the four broken turns; ACC5on all restart = all healthy; the
      no-spec reference picks spam in 6 of 8 turns.
    So "acceptance" in this loop was mostly the drafter's ability to predict post-EOS filler, and the
    "regime" was which filler the target picked at a near-tie after EOS — resolved randomly under
    the nondeterminism (findings 41–42: per request, prompt-independent) and deterministically
    once A and B were fixed (55, 58: fixed per depth, because the verify-shape numerics shift the
    tie). **The depth grid (58), the 40-turn DEG runs (9), the acceptance correlation (8), the
    "target moves under the drafter" story and every MTP ms/tok from `agentloop*.py` are
    contaminated by this.** The determinism findings (bit-level hashes, max_tokens 1–4) are not.
    What stands: bug A and bug B are real nondeterminism with validated fixes; "defect C" and the
    per-depth pattern are the artifact. `MTPGRID3` re-measures n=0..8 × 3 starts with the loop
    stopping at EOS. Raw: `notes/data/acclog_n2_cache_on.txt`, `acc_n2_cache_on_texts.json`,
    `acc_n5_cache_on_texts.json`, `mtpacc2-partial.txt`.

60. **59 holds at n=6 and with the cache off; eager mode with exact top-k is deterministic too.**
    `MTPACC2` continued: n=6 cache on — every broken turn is ~10 healthy steps (the answer) then a
    zero wall; n=2 cache off — the first four turns' step sequences are byte-identical to the
    cache-on run (the cache path plays no part). `DH5_a` (eager, `--enforce-eager`, exact top-k
    verified active, drafter hashes, 3 passes): per-turn acceptance identical to the decimal
    (71.7 / 75.0 / 72.1 / 66.0 / 66.0 / 60.6 / 68.7 / 60.0 %), every hooked drafter module
    identical across passes except one `embed_tokens` row at one draft call per turn (a call-
    alignment artefact of the grouping, not pursued — acceptance is identical). This replaces the
    withdrawn 56: eager mode has no extra nondeterministic kernel. Raw: `notes/data/mtpacc2.txt`,
    `acclog_n6_cache_on.txt`, `acclog_n2_cache_off.txt`, `acc_*_texts.json`.

61. **EOS-correct depth grid on the validated stack: no depth is broken; speculation does not pay
    on 30-token turns.** `MTPGRID3` (loop stops at EOS, align + exact top-k + det MoE, prefix cache
    on, compiled, 3 starts, `QSAFIX active` verified):

| n | s/turn (a/b/c) | real tokens | acceptance | accept len | counters a=b=c |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.09 / 2.08 / 2.07 | 237 | – % | – | yes |
| 1 | 2.63 / 2.64 / 2.67 | 208 | 74.4 % | 1.74 | yes |
| 2 | 2.72 / 2.65 / 2.63 | 227 | 66.0 % | 2.32 | yes |
| 3 | 2.70 / 2.67 / 2.67 | 227 | 57.0 % | 2.71 | yes |
| 4 | 2.55 / 2.58 / 2.56 | 166 | 40.9 % | 2.64 | yes |
| 5 | 2.66 / 2.66 / 2.67 | 166 | 34.2 % | 2.71 | yes |
| 6 | 2.61 / 2.56 / 2.54 | 145 | 36.1 % | 3.17 | yes |
| 7 | 2.78 / 2.56 / 2.58 | 149 | 30.3 % | 3.12 | yes |
| 8 | 2.70 / 2.70 / 2.70 | 142 | 24.8 % | 2.98 | yes |

all depths counter-identical across 3 starts: True

    Every depth reproduces its draft/accept counters across the three starts. Acceptance on real
    answer tokens falls from 74 % (n=1) to ~25 % (n=8) while the accept length rises to ~3 (n=6–7):
    the drafter is sound at every depth. Turn time is TTFT-bound at this answer length (~30–40
    tokens): 2.1 s/turn without speculation, 2.6–2.8 s with any MTP depth, because the one-block
    prefix-cache back-off costs more re-prefill than the decode saved (cf. #54713 for the Mamba
    side of that). The old-loop numbers (58) are retired; the served config's choice is a
    workload question, not a defect question. Raw: `notes/data/mtpgrid3.txt`.

62. **Prefill cost of the exact-top-k workaround: +6 % TTFT.** `MTPTTFT` (no spec, prefix cache
    off, det MoE on both arms, 3 starts each, 3 requests per point, per-arm activation verified):

    | prompt | stock `persistent_topk` | exact `torch.topk` (workaround) | cost |
    | --- | --- | --- | --- |
    | 7,503 tokens | 3.12 / 3.13 / 3.14 s | 3.32 / 3.32 / 3.44 s | +6.1 % |
    | 29,263 tokens | 11.25 / 11.31 / 11.38 s | 11.97 / 12.02 / 11.94 s | +5.8 % |

    The cost is flat in context length (the selection is O(rows × columns) either way), and far
    below the community overlay's −8…−40 % (which replaced the kernel on every layer with a
    Python top-k over the full logits *and* a canonical sort). The kernel fix (`patches/
    kernel-det/`) is meant to bring this to ≈0; its microbenchmark is queued. Decode cost of the
    workaround at c=1 is inside the run-to-run band (finding 61 vs 57). Raw: `notes/data/mtpttft.txt`.

63. **Kernel fix `patches/kernel-det` v2.3: correct and deterministic on sm_121; 1.3–4× the stock
    kernel's per-call time.** Built standalone as `_C_det` (three build defects fixed first, all in
    the standalone glue, none in the kernel: the stable-shim stream getter must be
    `aoti_torch_get_current_cuda_stream` behind `-DUSE_CUDA` — the generic `aoti_torch_get_current_stream`
    returns an opaque `c10::Stream` handle and segfaults in `cudaMemsetAsync`; the dynamic-smem request
    must leave room for the kernel's static `__shared__` BlockScan scratch, or `cudaFuncSetAttribute`
    rejects the full 99 KB opt-in). `test_det.py`: **177 / 177 cases** bit-identical across 6 (ALL-EQUAL:
    100, PIVOT-TIES: 20) calls and equal to the exact reference (value desc, index asc), on every path
    (decode ≤ 8k, medium ≤ 32k, multi-CTA > 32k). The stock op reproduced its own output in **0 / 177**
    cases (order) and on all 39 tie-heavy inputs returned a set other than lowest-index (a tie-break
    that is valid per call, but finding 53 showed it changes between calls). Raw: `patches/kernel-det/
    test_results.txt`.

    `bench_det.py` (5 × 50 launches, median µs, stock re-measured after det; full table in
    `patches/kernel-det/bench_results.txt`):

    | rows | n | k | stock | det | ratio |
    | --- | --- | --- | --- | --- | --- |
    | 1 | 1,024 | 512 | 8.3 | 10.4 | 1.26 |
    | 1 | 8,192 | 2,048 | 10.3 | 30.1 | 2.91 |
    | 1 | 32,768 | 2,048 | 18.5 | 71.8 | 3.87 |
    | 1 | 65,536 | 2,048 | 16.5 | 39.0 | 2.36 |
    | 64 | 8,192 | 2,048 | 18.5 | 57.5 | 3.10 |
    | 64 | 32,768 | 2,048 | 55.5 | 207.1 | 3.73 |
    | 64 | 65,536 | 2,048 | 108.6 | 231.9 | 2.13 |

    Two things follow. (a) The multi-CTA path (> 32k, deterministic emission on top of the existing
    radix rounds) is *cheaper* than the single-CTA `det_select_row` at 32k — the four rescans plus
    the in-block sort dominate; routing more rows to the multi-CTA path (`RADIX_THRESHOLD`) is the
    first optimisation to try. (b) Model-level cost estimate, 12 QSA layers: decode at 32k context
    +0.6 ms/step (≈ +1.5 % of 44 ms); prefill 7.5k tokens ≈ +55 ms (≈ +1.8 % TTFT, vs +6.1 % for the
    Python workaround in finding 62). Estimates until the end-to-end A/B runs. Note for the upstream
    PR: on GB10 the `num_rows > 32` filtered path is never taken (`sharedMemPerBlockOptin` = 101,376 <
    128 K), so the filtered-kernel change is compile-checked here but not executed.

    **Addendum (same day): `RADIX_THRESHOLD` 32768 → 16384 adopted** (`bench_results_threshold.txt`,
    177/177 tests again). Rows ≤ 16k are untouched; at 32k the multi-CTA path halves the det cost
    (1 row k=2048: 71.8 → 37.5 µs; 64 rows: 207 → 127 µs; only 24 rows regresses, 73 → 85). 8192 was
    also tried and loses at 64 rows × 16k (49 → 60 µs). Installed as `/opt/llm/kernel-det/_C_det.so`;
    `tools/determinism/qsadet_patch.py` (`VLLM_QSA_DET_TOPK=1`) routes the QSA selection to it for
    the end-to-end A/B. Not installed yet (queue: `mtpgrid0` running).

64. **PR #55122's test file: 70 / 70 pass against the built kernel, 70 / 70 fail against the stock op.**
    The three new pytest cases (`test_persistent_topk_deterministic`, `_all_equal`, `_pivot_ties`,
    `_narrow_value_range`; 70 parametrisations, 8 skipped for k ≥ n) run on the box with a plugin that
    swaps the op for `_C_det` (`patches/kernel-det/detplugin.py`): **70 passed** in 0.7 s. The same file
    against the venv's stock `_C.persistent_topk`: **70 failed** — every case, including the tie-free
    random rows, because the stock output order is not reproducible and not index-sorted. Raw:
    `patches/kernel-det/pytest_results.txt`. (Queue re-ordered the same afternoon: quick jobs first —
    pytest, prefill profile, HC-GEMM microbench — the grid remainder and the kernel A/B run from 22:00.)

76. **Third-party validation of the kernel (k3dani, GB10, 2026-09-03, PR #55122 comment 16:38 UTC):
    reproducible, throughput-neutral, quality-neutral.** RadixArk NVFP4, preview image, prefix cache +
    chunked prefill + PIECEWISE graphs + MTP=2, sequential requests, sha256 over top-20 logprobs per
    token: stock `persistent_topk` 0/4 prompts reproducible (10 distinct hashes each, diverging at token
    0); exact `torch.topk` and `VLLM_QSA_DET_TOPK=1` 4/4 on 512-token thinking runs, 3/4 on the 48-token
    runs where the one "failure" is a partial-prefix-cache-hit request whose logits differ from the
    full-hit path — the align-resume defect (#53798/#54076/#54173), not the kernel. Prefill throughput
    99–100 % of stock (exact fallback 87–90 %); decode unchanged. Batch invariance not provided (2–4
    identical concurrent requests differ from sequential and from each other; expected, GDN has no
    batch-invariant path). MoE fused finalize: 0 divergences across 80 warm requests on their shapes
    (shape-dependent; a data point for #54945). Quality, 50-item Hungarian KIE suite × 3 runs:
    **95/100 with 0/50 unstable items** vs stock 97/100 with **13/50 unstable**; the one-point gap is a
    single near-tie reasoning fork. Artefacts: k3net/docai-evals, experiments/2026-09-03-qwen38-flash-next-det-topk-kernel-batch-invariance-gb10.

80. **The preview build lacks vllm#50729 (overlapping Mamba state-copy race, merged 2026-08-17).**
    Found via blazux/qwen3.8-Flash-DGX (their image carries it plus a bounds guard by Saren-Arterius):
    none of the fix's added lines exist in `vllm-venv-fnext`'s `v1/worker/mamba_utils.py`; all are in
    the main venv. It is on our prefix-cache path (conv-state left shifts at align boundaries). Our
    bit-identity results for cold vs partial-hit were measured without it, so they were either lucky or
    the race needs concurrency we did not run; k3dani's 3/4 partial-hit case (finding 76) is a
    candidate. The upstream diff applies cleanly to the preview (4 hunks, offsets 5–6); queued
    (`race50729`) for after the night chain, backup `mamba_utils.py.pre50729`, diff saved as
    `patches/upstream-candidates/vllm-pr50729-mamba-state-copy-race.diff`. Also in blazux's image and
    worth adopting: the FLA shared-memory gate (sm_121 has 99 KiB, the gate asks 100 KiB → small tiles
    for every gated GDN kernel) and the `chunk_delta_h` `num_warps=2` pin (fla#953 tl.dot race on
    Blackwell) — `tools/main/fla_gb10_patch.py`, before/after bench queued (`flagate`).
    Upstream merged since the preview fork that the main build has and the preview does not (our
    files only): #52789 internal prefill checkpoints for Mamba prefix caching (9–25 % TTFT claimed),
    #53388 disabling the trailing prefix-cache block drop under spec decode (= plan §5 item 1),
    #53877 GDN decode beta in FP32, #53456 XD-RoPE grid on prefix hit, #54251 GDN RMSNorm warm-up.

81. **Grid start c (stock preview stack, EOS-correct agent loop, 22:23): the MTP acceptance ladder is
    the same for the third time.** Rate by n: 60 / 58 / 46 / 40 / 37 / 29 / 26 % for n = 2..8 (a/b:
    64/65, 45/50, 47/49, 37/41, 39/30, 27, 25); mean accepted length rises 2.2 → 3.0 and plateaus from
    n=5; seconds per turn: n=0 1.90, n=2 2.12, n=3..7 2.36–2.49, n=8 2.78 (raw: `notes/data/mtpgrid0c.txt`,
    table: `notes/data/mtpgrid0-partial.md`). Three starts agree within the run-to-run band of finding 61,
    so the before-column for the #53142 correction is complete except n=1 (redo queued) — and the
    conclusion holds: on the preview stack spec decode does not buy the agent loop a faster turn at
    130–230-token turns; the no-spec arm is the fastest per turn at every start. Three arms of this run
    died for two reasons that are both infrastructure, now fixed and documented in `failure-modes.md`
    (the swapfile the reboot dropped; the PSI guard's "some" rule).

82. **Server-level A/B of the deterministic top-k kernel (`kdetab`, 22:24–01:24, three starts per arm,
    stock preview stack, det-finalize/qsafix inert, align patch on both arms; raw `notes/data/kdetab.txt`).**
    TTFT is unchanged: 8k (7,503 tok) median stock 3.10 / 3.29 / 3.10 s vs det 3.12 / 3.35 / 3.12 s;
    30k (29,263 tok) stock 11.24 / 11.55 / 11.26 s vs det 11.22 / 11.62 / 11.27 s — every pair inside the
    start-to-start band, activation line (`QSADET active: _C_det.so`) present on every det arm, zero patch
    lines on every stock arm. Decode, MTP n=5, 8-turn EOS-correct agent loop, seconds per turn: stock
    2.70 / 2.68 / 2.68, det 2.69 / 2.54 / 2.66, exact-top-k 2.71 / 2.65 / 2.64. The det and exact arms
    produced the **same run six times out of six** (166 tokens, 65 drafts, 111 accepted, 34.2 %,
    length 2.71) — the determinism property at the server level, across restarts, and evidence that the
    kernel selects the exact set; the stock arm's differing acceptance (39.5 / 37.1 / 38.0 %) is a
    different trajectory (208–224 tokens), not a kernel cost. Same for the ms/token gap (stock ~100,
    det ~127): fewer tokens per turn under the same per-turn fixed cost, s/turn is the like-for-like
    number. Conclusion for PR #55122: no TTFT and no per-turn cost end-to-end on GB10.

84. **`_C_det` rebuilt from the PR #55122 follow-up source (per-call device properties, k=1024 test
    shapes) — 177/177 tests pass, installed 02:21** (`notes/data/kdetrebuild.txt`). The box copy now
    matches the PR head `cbd0642a`; the kdetab A/B (finding 82) ran on the previous build, whose kernel
    body is identical (the follow-up only moved the smem query per call).

87. **Grid complete: three starts for every n = 0..8 on the stock preview stack** (redo arms
    `notes/data/s7redo.txt`: S7_b 29.9 % / 3.09 / 2.89 s, S8_b 32.2 % / 3.58 / 2.89 s, S1_c 71.2 % / 1.71 /
    2.24 s; table `notes/data/mtpgrid0-partial.md`, #53142 follow-up draft updated). The ladder over
    three starts: n=1 72 / 73 / 71 %, n=2 64 / 65 / 60, n=3 45 / 50 / 58, n=4 47 / 49 / 46, n=5 37 / 41 / 40,
    n=6 39 / 30 / 37, n=7 27 / 30 / 29, n=8 25 / 32 / 26. Seconds per turn never beat the no-spec arm
    (1.85–2.09) on this stack. Infrastructure notes from the redo: the swapfile fix holds (both arms that
    died twice loaded and ran); the headroom guard in the redo runner sat *before* the previous arm's
    stop, so it only added a 10-min wait per arm — fixed in the runner afterwards, not a measurement issue.
88. **vllm#50729 (overlapping Mamba state-copy race) applied to the preview venv 03:21**, all four hunks
    at offsets 5–6, `py_compile` clean, backup `mamba_utils.py.pre50729` (`notes/data/race50729.txt`).
    Every preview measurement from here on carries it; nothing above was re-measured with it.

93. **The main build had never served with MTP: the MTP head's `lm_head` is built unquantized** (`Qwen4ExpMTP`
    builds its own `ParallelLMHead` without `quant_config`, and `_map_mtp_name` maps the checkpoint's
    `lm_head.weight_scale_inv` onto it → "no parameter named lm_head.weight_scale_inv"). Same defect as
    the body head (patched in `lmhead_patch.py`), same fix: `tools/main/lmhead_mtp_patch.py`. The
    trailing-block A/B (`blockdrop`) lost its first arm to it and was stopped; `blockdrop2` applies the
    patch in preflight and re-runs all six arms after `hcbench2`.
    **Second layer of the same defect (11:22):** with the head quantized, the checkpoint's `lm_head.weight_scale_inv`
    still has no home — the MTP loader lacks the body's FP8_PB_WO rename/reshape (`scaleinv_patch.py`);
    `tools/main/scaleinv_mtp_patch.py` adds it to `Qwen4ExpMTP.load_weights`. `blockdrop3` applies both. Cost of
    finding it in two rounds: ~1 h of dead arms (four in `blockdrop2`, `notes/data/blockdrop2-failed.txt`).

## Independent corroboration

[vllm#54173](https://github.com/vllm-project/vllm/issues/54173) — open, different reporter, **same
model and same GB10 sm_121 hardware** — independently reports both of our findings: prefix caching
implicated, *and* "nondeterministic greedy decoding above ~2K tokens with prefix caching disabled".
It also reports `mamba_cache_mode` "align" and "all" failing identically, which predicts our queued
`M_all` vs `M_align` discriminator will come back null.

[vllm#47861](https://github.com/vllm-project/vllm/pull/47861) fixed MTP + prefix caching correctness
for hybrid Mamba models (tool-call leakage, needle recall failures, ~20% accuracy drops). It was
**closed unmerged**; only its scheduler half landed via #51113, which **is** present in our build
(`_mamba_block_aligned_split`, `mamba_partial_cache_hit`). The unmerged half — *don't apply
EAGLE/MTP peek-and-drop to Mamba state groups, since recurrent snapshots cannot be rewound* — is a
live candidate for what we are seeing.

## Narrowed to

The align-mode state machinery, which runs regardless of hits: copy-on-write into private blocks
(`_producer_partial_tail_reqs`, `last_state_block_idx`) and `postprocess_mamba_align_gpu`, a fused
GPU postprocess that mixes state copies with accepted-token updates **without a CPU-GPU sync**.
Plus whatever drives the separate generation-side path.

## Refuted this session (kept so they are not re-run)

- PLE gather as the source — bit-identical output in both bisection groups
- PLE FP8 global scale destroying rows — 0 dead rows, uniform magnitudes
- Language-dependent damage via the n-gram hash — the hash *is* token-ID dependent, but all table
  regions quantise equally well, so there is no bad region to land in
- Off-by-one / missing bounds masks in the model-specific Triton kernels
- Stale `blocks_buffer` tail reaching the expansion kernel
- "Prefix caching is necessary for the divergence" — true for prefill only; generation diverges
  without it

## Instruments (`tools/determinism/`)

| script | what it answers | needs |
| --- | --- | --- |
| `layerhash_patch.py` | which layer first differs; includes an async **race detector** (hash twice with a sync between) | server, `--enforce-eager` |
| `topk_boundary.py` | off-by-one / reads-past-bound in `persistent_topk`, vs a `torch.topk` oracle | idle GPU, seconds |
| `kernelbox_capture.py` / `_replay.py` | kernel determinism **and purity** (`replay(input) == captured output`) | real request, then fresh process |
| `kernelbox_adversarial.py` | rare events (rate bound, not a boolean), out-of-bounds reads via poison padding, alignment sensitivity | a capture |
| `runtime_determinism.py` | CUDA/driver-level nondeterminism, incl. allocator churn | idle GPU |
| `sigcompare.py` | separates "unstable" from "systematically wrong" across arms | results files |
| `accepcorr_report.py` | does `ms/tok` track `mean_accept_len` across restarts | results file |

## Method rules this cost us

- Never quote an MTP timing from one start: three minimum, report the range, treat <2x as
  unresolved. → `failure-modes.md`
- Group passes by an input fingerprint (layer-0 hash) before comparing layers. Comparing
  unmatched passes said "layer 0 already differs", which would have pointed at embeddings.
- One flag can change two things: `--no-enable-prefix-caching` also flips
  `mamba_cache_mode` align -> none.
- A negative result is only as strong as its sample size — report the bound, not "clean".

109. **Field, vllm#54928 (Windless84, 2026-09-04 16:45): the block-verification forward ranks a different token than the
    single-token forward on a dense Qwen3.8-27B-FP8, stock vLLM 0.28.0, RTX PRO 6000 (sm_120).** Verifier-side attribution
    at the first divergence in every instrumented run: E == V != A (emitted = verifier argmax ≠ target-only argmax) at
    positions where the target-only top-2 gap was ≤ 0.125 nats; `--enforce-eager` does not remove it; custom RMSNorm
    makes it worse and non-repeatable across launches; the target alone is not batch-shape invariant (`--max-num-seqs`
    1 vs 4 diverges at 188, with a concurrent request at 69); `VLLM_BATCH_INVARIANT=1` is refused for GDN. Same pattern on
    a patched NVFP4 build (first divergence 12). This is our diagnostic 2 from the 09-04 reply run by someone else, with
    the same outcome: the drafter exposes a batch-shape dependence of the target. On GB10 the same signature came from
    three concrete kernels (findings 80–93); on their stack (FP8 dense, no prefix cache, sm_120) none of those three
    applies, so the next candidate is the GEMM itself: the sm120 CUTLASS FP8 dispatch and cuBLAS both choose kernels by
    M (the swap-AB / small-M path we hit in PR #55180's test oracle), so an M=1 decode row and the same row inside an M=8
    verification block need not be bit-identical. `tools/gemm_m_invariance.py` (runner `gemminv`, queued behind the
    union runs) measures exactly that on the model's dense shapes for per-channel FP8, blockwise FP8 and BF16 cuBLAS.
    Reply drafted (`notes/upstream/comment-54928-reply-windless.md`), to be posted with the GEMM result after a go.

112. **GEMM batch-shape invariance on sm_121 (`gemminv`, `tools/gemm_m_invariance.py`, `notes/data/gemminv.txt`):
    per-channel FP8 is bit-identical across M, blockwise FP8 and BF16 cuBLAS are not.** Same first row through
    M = 1, 2, 3, 4, 8, 9, 16, 32, 64, 128, 256, 1024, 4096 on the dense shapes 12288×2560, 5120×5120, 16384×2560:

    | path | row 0 == M=1 result | max |diff| (bf16 out) |
    | --- | --- | --- |
    | per-channel FP8, `cutlass_scaled_mm` (sm120 dispatch) | every M | 0 |
    | blockwise FP8 128×128, `cutlass_scaled_mm` (sm120 blockwise, the #55180 kernel) | M=1 only; differs from M=2 on | 1.0–1.5e-2 |
    | BF16 cuBLAS | M=1 only; differs from M=2 on | 0.25–2.0 (unscaled randn) |

    So on this GPU a decode row at M=1 and the same row inside an M=8 verification block do not go through the same
    arithmetic in any blockwise-FP8 or BF16 dense projection; only the per-channel FP8 path is M-invariant. Qwen's
    official `Qwen3.8-27B-FP8` is blockwise, i.e. Windless84's stack (finding 109) has this in every dense layer
    of every GDN and QSA block; the 0.1-nat flips they attribute are of the order such 1-ulp output differences
    produce after 40+ layers. It is a candidate, not an attribution: the test is on the GEMM alone, not on the
    logits. Reply draft filled (`notes/upstream/comment-54928-reply-windless.md`); a logit-level version would run
    the target with `--max-num-seqs 1` and top-5 logprobs at q_len 1 vs 8 through the same prefix — which is what
    their instrumented run already did, with the same outcome.

116. **Field, vllm#54521 (davidcanar, 2026-09-04 21:19): on gfx1151 the divergence survives strictly sequential, idle,
    prefix-cache-off runs (5/5 distinct at every length), so an order-nondeterministic kernel is in their path; the
    batch-shape channel is not their mechanism.** They corrected their own earlier claim (their determinism harnesses
    had prefix caching on; the re-run without it holds). Narrowing by our criterion: MoE combine excluded bitwise
    (`moe_sum` is inside the op they tested), the Triton ragged sparse-MLA decode has no atomics (two-stage split-KV),
    the collective is the open one (`NCCL_PROTO` unpinned — next). Two new facts: (a) the outcome set is *discrete* —
    specific output hashes recur across independent runs — which fits a reduction-order variable and not garbage reads;
    (b) `--max-num-seqs 1` hard-faults the engine in `_deepgemm_fp8_paged_mqa_logits` (the GLM sparse indexer's paged
    MQA logits kernel) on the first request — an indexing bug that faults only when the access leaves a mapped page
    would be consistent with silent wrong reads at larger max_num_seqs. Their gfx1151 rows for our M-invariance table:
    ROCm BF16 (hipBLASLt) row 0 is M-invariant for M = 1…64 (1…32 on the narrowest shape) and switches kernel at
    M ≥ 128; the int4 W4A16 Triton fused MoE is bit-identical at every M despite per-M tuned tiles. So on their stack the
    E == V ≠ A channel of #54928 is inactive at MTP verification widths, unlike sm_121 where BF16 cuBLAS and blockwise
    FP8 differ from M = 2. They offer `tools/gemm_m_invariance_rocm.py` as a PR to our repo. Reply drafted
    (`notes/upstream/comment-54521-davidcanar-2.md`): accept the PR, the recurring-hash argument, the fault deserves its
    own issue, and the GB10 two-node comparison offer stands if the collective is the channel.

122. **Finding 112's blockwise-FP8 row is WITHDRAWN — a scale-layout artifact of our own script (found by jahnclawdmonet on
    #54521, 2026-09-05 07:28, on an RTX PRO 6000).** `gemm_m_invariance.py` v1 passed the activation scales row-major
    `[m, K/128]` and the weight scales row-major `[K/128, N/128]`; the sm120 CUTLASS blockwise kernel reads no strides
    and deduces the layouts from the shapes (M-major for A, K-major for B), so the cell computed a different GEMM whose
    row 0 depends on m — 53–126 ulp off the intended product at every M, per their float64 check. With the production
    layouts (column-major activation scales, weight scales built `[N/128, K/128]` and passed transposed) **blockwise FP8
    row 0 is bit-identical at every M on sm_120**; per-channel FP8 (identical) and BF16 cuBLAS (1 ulp, M-dependent) stand.
    Their fixed-M repeat check (36 cells × 30 calls, three processes) found no call-to-call divergence in any dense path.
    Consequences: (1) `tools/gemm_m_invariance.py` v2 uses the production layouts; rerun on the GB10 queued (`gemminv2`)
    to replace finding 112's table; (2) our #54521 reply of 23:4x (posting 29) carried the wrong row — correction to post;
    (3) the unposted #54928 draft loses its blockwise sentence: on this evidence the E == V ≠ A channel on an FP8-dense
    stack cannot come from the blockwise GEMM, only from BF16 cuBLAS paths (1 ulp) or elsewhere. Lesson for the harness:
    a `stride(0) == 1` assertion on 2-D scales, and a float64 reference on the first cell before any table is quoted.

    **GB10 rerun (`gemminv2`, `notes/data/gemminv2.txt`, script v2):** blockwise FP8 row 0 bit-identical at every M on
    all three shapes; per-channel FP8 identical; BF16 cuBLAS differs from M = 2 on (0.25–2.0 unscaled). Finding 112's
    table is replaced by this one; the withdrawn row is confirmed an artifact on sm_121 as on sm_120.


127. **MTP corrupts the output of every request but one when two or more prompts are prefilled in the same step — the
    production build, prefix cache on or off, temperature 0; clean without speculation and clean whenever the prompts
    arrive 150 ms apart (`acceptcell2/3`, `notes/data/acceptcell2*.{txt,jsonl}`, `acceptcell3*`).** This is what
    finding 126 called a "drafter collapse": the corrupted request's own text is garbage from its second token on
    (`"Basedsumsumsumsum…"`, mixed-script noise; the first token is the right one), so the target's state is wrong,
    and the 9 % draft acceptance is the consequence. Probe: c concurrent chat requests, identical 549-token prompt up
    to a salt, 128 new tokens, `temperature 0`, `ignore_eos`; a request is "corrupted" if its text repeats `sum`,
    is > 5 % non-ASCII, or has < 40 % distinct words (the flag agreed with the acceptance flag on all 60 cells).

    | build | config | cells with ≥ 1 corrupted request | corrupted per collapsed cell |
    |---|---|---|---|
    | dev401 + our overlay (union disabled), MTP n=3, cache on | c = 1 / 2 / 3 / 4 / 5 / 8, 10 cells each | 0/10, 3/10, 4/10, 4/10, 4/10, 5/10 | 1 / 2 / 3 / 3 / 3 (never all) |
    | dev401, MTP n=3, GB10 table, cache on (126) | c=4 ×3 | 1/3 | 3 |
    | dev401, MTP n=3, cache **off** (126) | c=4 ×3 | 1/3 | 3 |
    | **production dev352, no overlay, MTP n=3, cache on** | c=2 ×10, c=3 ×6 | **5/10, 2/6** | 1, 2 |
    | production dev352, MTP n=3, staggered pairs | delay 0 / 0.15 / 0.4 / 1.0 / 3.0 s, 6 each | **3/6, 0/6, 0/6, 0/6, 0/6** | 1 |
    | production dev352, **no speculation** | c=2 ×10, c=3 ×6, pairs at all five delays | **0/46** | — |

    Reading: (1) the trigger is a scheduler step that prefills two or more requests; with speculation off the same
    batches are clean, and with 150 ms between arrivals (the second prompt joins as a prefill next to the first
    request's decode) they are clean too. Which request survives varies (A or B); the number corrupted per cell is
    c−1 up to three and then stays at three at c = 5 and 8, where the KV budget lets five requests run and the
    late ones join singly. (2) Not our code: the production venv (dev352, no overlay) shows it at the same rate.
    Not the prefix cache, not the split-K table, not preemption (counter 0), nothing in the log. Not prompt-length
    padding: every prompt is 549 tokens. (3) The nightly's c=16 cells were healthy because the old probe started
    threads sequentially, so the first prefill ran alone — a probe artifact that hid the defect all week. (4) Same
    disease as vllm#55357 (tgmerritt, 2× RTX PRO 6000, NVFP4 checkpoint, MTP n=3: episodic 0 % acceptance with
    degenerate output for the life of a request, hours long); they lack a reproducer and suspect the prefix cache —
    our cache-off run rules that out. **Impact: every MTP-on serve on this box, including the production
    configuration, garbles requests whose prefill shares a step; agent clients that fire parallel tool calls hit
    exactly that.** Mitigation until fixed: serve without MTP, or serialise prefills. Bisect running
    (`acceptcell4/5`): equal vs unequal prompt lengths, MTP n=1, `--enforce-eager`, `index_share_for_mtp_iteration`
    off → finding 128.

128. **Bisect of the multi-prefill MTP corruption (`acceptcell4/5`, `notes/data/acceptcell4.txt`, `acceptcell5.txt`): none of
    prompt length, draft depth, CUDA graphs or the shared step-0 selection matters — the drafter's prefill step over
    two prompts is sufficient.** Production venv (dev352), simultaneous pairs, 128 tokens, temperature 0:

    | arm | corrupted |
    |---|---|
    | equal length (543/543), one token apart (544/545), 34 apart (545/511), 100 apart (545/443), equal again | 3/8, 2/8, 3/8, 4/8, 3/8 |
    | MTP **n=1** (no multi-step draft decode) | 2/10 |
    | MTP n=3, `--enforce-eager` | 3/10 |
    | MTP n=3, `index_share_for_mtp_iteration=false` | 3/10 |
    | control, MTP n=3 (127) | 5/10 |

    Reading: the draft's prefill over a multi-request batch, reusing the target's attention metadata and slot mappings
    (`AutoRegressiveSpeculator._prefill`), corrupts the state of all but one of those requests; the decode steps, the
    graph mode and the top-k reuse are innocent. A second opinion (2026-09-05) sharpened the next cut: not "skip every
    store" but (a) force the draft indexer onto the unfused reference path, and (b) withhold only the raw 8-row ring
    and its RoPE tail during the drafter's prefill (compressed keys, top-k and attention untouched), through the
    prefill lifecycle hooks; the fused pre-indexer commits the ring inside `qsa_pre_indexer.py`, so a patch on the
    visible `qsa_store_cache_rows` calls would have gated nothing. Both arms are queued (`acceptcell7`), behind a
    batch-composition log run (`acceptcell6`). Candidate mechanism per that review: slot/metadata ownership of the
    second request in the reused metadata; the store kernels themselves carry no request notion and can only execute
    a wrong slot mapping faithfully.

129. **Batch-composition log (`acceptcell6`, `notes/data/acceptcell6.txt`, `acceptcell6-BL.jsonl`; one env-gated log line
    in the V2 runner, request ids sent explicitly): the corruption is deterministic on the step's shape. A step whose
    batch consists of prefills only corrupts every prefill after batch row 0 (5/5 such steps: `[545,545]` → row 1
    dead, three times; `[545,545,545]` → rows 1 and 2 dead, twice). A step with a speculating decode row in front and
    two prefills behind it (`[4,545,545]`, 3/3) is clean, and prefills in separate steps are clean (8/8).** Overlay
    venv, MTP n=3, 16 cells. This is why the earlier c=16 cells looked healthy (their first prefill ran alone and every
    later prefill joined behind decode rows) and why the stagger of 150 ms cures it. Whether the two-prefill step is
    poisoned by the drafter's writes or by the target's own state handling under the speculative configuration (ring
    capacity 8 instead of 4, spec-sized conv-state windows) is what the queued arms separate: draft indexer unfused
    (`U1`), draft-prefill ring commit withheld (`R1`), and drafter never run at all with the spec config on (`ND`).

130. **The drafter is innocent; the defect is a strided state-index view fed to a stride-blind kernel (`acceptcell7/8`,
    `notes/data/acceptcell7.txt`, `acceptcell8.txt`).** Three arms on the overlay venv, simultaneous pairs, MTP n=3:
    draft indexer forced onto the unfused reference path — **3/10** corrupted; draft-prefill commit of the raw ring and
    RoPE tail withheld through the prefill lifecycle hooks — **2/10**; **drafter never executed** (zero drafts returned
    before any draft forward, acceptance 0 % throughout) — **1/10 at c=2** plus the c=3 cells clean. So the corruption
    needs only the speculative *configuration* and a prefill-only step with ≥2 requests; nothing the drafter computes
    or writes is involved.

    Mechanism (code, both venvs — dev352 and dev401): with speculation configured, the Mamba-style block table has
    `1 + num_speculative_blocks` columns per request (`mamba_get_block_table_tensor`, mode "none"). The base Mamba
    metadata builder takes the prefill state indices as the first column, `state_indices_tensor_p = state_indices_tensor_p[:, 0]`
    (`vllm/v1/attention/backends/mamba_attn.py:570`), a **strided view**. The PLE short-conv kernels load the
    per-request state slot as `sid = tl.load(state_idx_ptr + r)` (`ops/ple.py:389`, `:504`), unit stride assumed, so
    prefill row r ≥ 1 resolves `block_table[0, r]` — request 0's speculative checkpoint blocks — instead of
    `block_table[r, 0]`. The prefill writes its conv state there; request 0's own speculative steps overwrite those
    blocks; the request's decode reads its correct block, never written. Row 0 is right; rows 1..3 die (four columns);
    row 4 lands in request 1's unused primary block and survives — the 3/2 split seen at c=5 (127). Without
    speculation the table has one column and the view is contiguous; with a speculating decode row in the batch the
    spec branch gathers indices by advanced indexing (contiguous) — the two clean cases of 129. GDN is unaffected:
    `causal_conv1d_fn` takes `stride_cache_indices`, and the recurrent state is indexed through torch. The same
    stride-blind load exists in the other short-conv consumers of `state_indices_tensor_p` and deserves a check.
    **Fix candidate under test (`acceptcell9`): `.contiguous()` on the state-index vector before the PLE conv
    kernels, drafter on; expected 0 corrupted of 20 cells.**

131. **Fix confirmed (`acceptcell9`, `notes/data/acceptcell9.txt`): with the state-index vector made contiguous before
    the PLE short-conv kernels, 20/20 cells are clean with the drafter on — c=2 ×10, c=3 ×6, c=5 ×4, simultaneous
    arrival, temperature 0 — against 30–50 % corrupted cells stock. The log line added for the run shows the vector
    arriving with stride 4 (= 1 + `num_speculative_tokens`) in both `prefill` and `decode` mode, so non-speculating
    decode rows after row 0 were exposed the same way.** Upstream form of the fix: the two kernels take
    `state_idx_stride` and index `state_idx_ptr + r * state_idx_stride` (branch
    `fix/mamba-prefill-state-indices-contiguous`, one file + a regression test that runs the prefill store with a
    strided column view of a 2- and 4-column table against the contiguous copy). MTP c ≥ 2 numbers become quotable
    again once the serving venvs carry the patch; production can re-enable MTP with the one-file overlay.

    **Upstream (2026-09-05 16:xx):** the identical fix was already open as vllm#55375 (peakcrosser7, opened 2026-09-04, same two
    loads, `state_idx_stride = state_indices.stride(0)`, maintainer-approved, CI running) — found after ours was posted. #55467 closed as
    duplicate; the GB10 evidence (no-drafter arm, row mapping, 0/20, strided decode-mode test) is on #55375. Their test covers the strided
    prefill path only; our decode-mode case stays on the branch as a follow-up offer.

133. **vllm#53051 (prefill of exactly 1 + n_spec tokens misclassified as uniform decode → GDN state loss) does not
    reproduce on the GB10 with Model Runner V2 (`bug53051`, `notes/data/bug53051*.txt/jsonl`).** MTP n=3 versus no
    speculation, same token-id prompts via `/v1/completions`, temperature 0, 48 new tokens, two repetitions each, prompt
    lengths 3, 4 (= 1 + n_spec), 5, 6, 8 and chunk tails 4095–4100 and 8191–8193 (max_num_batched_tokens 4096):
    every output coherent, no degenerate text at any length, and the ≥ 4095-token prompts byte-identical between
    repetitions and between the MTP and the no-spec server in 6 of 8 lengths. By construction: the V2 runner's
    `get_uniform_decode_token_count` requires `has_prefill == False`, and a request is prefilling while
    `num_computed_prefill_tokens < prefill_len` (`vllm/v1/worker/gpu/model_runner.py`), so a 4-token prompt or a
    4-token chunk tail can never be dispatched as a uniform decode; the V1 runner's shape-only `_is_uniform_decode`
    is the affected path, fixed by #53059. Side observation, not this bug: the 3–8-token prompts diverge between
    repetitions at temperature 0 in *both* arms from the second token on, and 4096/4098 diverge in one arm — the
    near-tie nondeterminism of finding 116 / #54521, visible here because a 3-token prompt has no context to break
    ties. Draft for the thread: `notes/upstream/comment-53051-gb10-v2.md` (needs go).

    **vllm#55375 merged 2026-09-05 14:02 UTC.** Nightlies after that carry the fix; the overlay venv carries it as `ops/ple.py.orig-dev401`-backed overlay; the production venv (dev352) still needs the one-file overlay or a nightly bump.

134. **c=1 MTP-3 decode under the profiler: the GPU is busy 73.5 ms of a 159.6 ms step; the PLE offload handshake is
    fully hidden; the idle is ~2,600 launches per step in PIECEWISE cudagraph mode (`plewait`, overlay venv, record_shapes,
    28 steps, `notes/data/plewait.txt`, trace on the box).** Per step: kernel busy 73.5 ms, idle 85.7 ms (medians);
    kernels per step 2,636; the gap immediately before the first PLE kernel (`_ple_conv_kernel`, `_ple_gate_kernel`,
    0.3 ms each per 28 steps) is **0.00 ms** — the host gather + H2D is entirely behind the embedding and layers 0–1,
    so PLE prefetch under speculation has nothing to buy. The largest single gaps sit before
    `triton_poi_fused__to_copy_embedding_repeat_0` (the first kernel of a forward; 2.3 ms per step = sampler/scheduler
    host time); the rest of the idle is spread thinly between launches — the server runs
    `cudagraph_mode=PIECEWISE` with capture sizes [1,2,4,8], so ~110 compiled pieces per step are called from Python
    (`## Call CompiledFxGraph ##` ×109/step) and every piece boundary pays host time. Caveat: the profiler itself adds
    per-op host overhead, so the 46 % duty cycle is an upper bound on the idle; the unprofiled step time from the `dv`
    run (tok/s and accept length at c=1) decides how much is real, and the `cg` A/B (PIECEWISE vs FULL_AND_PIECEWISE vs
    FULL_DECODE_ONLY) measures the lever directly. GPU-time budget inside the step: blockwise-FP8 dense GEMMs 31.8 %
    (2,804 calls at 237 µs each; at M=4 the ~25 MiB projections have a ~96 µs byte floor → ~2.5× above it, the largest
    kernel-level waste in decode); MoE grouped GEMMs 32 % (at their expert-byte floor, finding 137); BF16 GEMMs on
    unquantized weights 16.5 % (shared expert 2560→1280 + 640→2560 = 72 ms, hyper-connection low-rank 42 ms, router
    20 ms, MTP-layer dense ~25 ms per 28 steps); GDN update 2 %; QSA 0.7 %. The c=4 profile was lost: `/stop_profile`
    took longer than the client's 900 s timeout to export the shaped trace.


135. **Reduced-vocabulary MTP drafting on the FP8 head: +6.4–6.8 % single-stream decode at every slice size, +2–3 % at
    c=4, acceptance unchanged at 32k (`dv`, three server starts × four arms, 8 real held-out agent prompts at c=1 and 3×4 at
    c=4 per start, MTP-3, `notes/data/dv.txt`, per-request JSONL in `notes/data/dv/`).** The drafter's argmax runs over an
    exact BF16 dequant of the FP8_PB_WO head rows listed in `FN_DRAFT_VOCAB` (`tools/draft_vocab/dv_patch.py`, hooked
    into the V2 speculator's `_validate_local_argmax_reduction`, engaged via `use_local_argmax_reduction`); the vocabulary
    is frequency-ranked on our own agent output (finding: `notes/data/draft-vocab-coverage.txt`; held-out coverage 95.5 /
    98.1 / 99.6 % at 8k / 16k / 32k). Draft head per draft step: 606 MiB → 40 / 80 / 160 MiB.

    | arm | c=1 tok/s median (3 starts) | paired Δ vs full | Δ acceptance | wins | c=4 paired Δ |
    | --- | --- | --- | --- | --- | --- |
    | full head (local argmax) | 25.4 (25.3 / 25.6 / 25.4) | — | 45.5 % | — | — |
    | 8k | 27.1 | **+6.8 %** | −1.9 pp | 22/24 | +2.2 % |
    | 16k | 27.3 | **+6.8 %** | −1.1 pp | 21/24 | +2.9 % |
    | 32k | 27.4 | **+6.4 %** | +0.5 pp | 20/24 | +3.0 % |

    Wall time includes ~1.7 s of prefill per 5k-token prompt, so the decode-only gain is ~8–9 %, matching the byte model
    (3 × ~0.5 GB less per step at 273 GB/s on a ~72 ms step). Size does not matter for speed because even 32k (160 MiB) is
    small against the 606 MiB it replaces; it matters for acceptance only below 16k. **Pick 32k for prod**: same gain, no
    acceptance loss, 99.6 % coverage and the most headroom against traffic drift. Baseline on this traffic, for the record:
    acceptance 41–50 % of draft tokens, accept length 2.2–2.5 — the MTP head is weak on agent output, which puts drafter
    quality (online fine-tune) above further byte shaving. Start-to-start noise on identical configs was ~5 % (the first,
    hook-less run gave two full-head starts: 25.0 vs 26.2 tok/s), so the +6–7 % with 20–22 of 24 paired wins is above it.
    One garbage flag (16k, start 1, c=4) was a false positive: coherent reasoning text repeating "Let me look at …".
    Lossless by construction — the verifier sees the full head; the slice only changes proposals.


136. **cudagraph_mode is not a decode lever here: PIECEWISE, FULL_AND_PIECEWISE and FULL_DECODE_ONLY give the same MTP-3
    decode on real agent prompts (`cg`, one start per arm, `notes/data/cg.txt`).** c=1 medians 26.5 / 26.2 / 26.5 tok/s,
    c=4 46.9 / 43.4 / 45.4 tok/s (aggregate), acceptance 48–51 % in all arms — every difference inside the 5 % start-to-start
    noise. The config echoed each mode, but the logs carry no capture messages for any arm, so whether the FULL modes
    actually captured the hybrid GDN/QSA + PLE-offload forward or fell back silently is not established; either way the
    dial does nothing as shipped. Together with the unprofiled step time (≈ AL / tok/s ≈ 72 ms at c=1, equal to the
    73.5 ms of GPU-busy time in det-134) this closes the "46 % idle" question: it was profiler overhead, the c=1 decode
    step is GPU-bound. Remaining decode levers in order: the small-M blockwise-FP8 GEMM (32 % of the step at 2.5× its byte
    floor), the draft-vocab slice (det-135, measured +6–7 %), FP8 for the BF16 leftovers (shared expert, hc, router, MTP
    dense; 16.5 %), and drafter quality (acceptance 41–50 % on agent traffic).


137. **Draft-vocab slice, decode-only (streaming, TTFT subtracted): +6 % per stream at c=1 and c=8, +4 % at c=4, two starts
    per arm (`dvrate`, 32k slice vs full head, MTP-3, 400 output tokens on the held-out agent prompts, `notes/data/dvrate.txt`,
    per-request JSONL in `notes/data/dvrate/`).** Answers "what tok/s": single-stream decode on this traffic is **38.3 tok/s
    with the full head and 40.5 tok/s with the 32k slice** (per-start medians 38.9/38.3 vs 41.7/40.0); c=4 21.9 → 23.0 per
    stream (88 → 92 aggregate); c=8 19.8 → 21.0 (158 → 168 aggregate). Paired by prompt: +5.8 % mean at c=1 (10/12 wins),
    +3.7 % at c=4 (5/6), +6.2 % at c=8 (4/4); acceptance −2.6 pp at c=1, flat at c=4/8. TTFT of the 5k prompts is 2.1 s at
    c=1 and unchanged by the slice. Consistent with det-135's wall-clock +6.4 % (which included that prefill). The c=8 TTFT
    of 26 s is eight cold 5k prefills serialised at 16k chunks — a scheduling artefact of the cell, not a decode effect.


138. **ZC502's position-resolved regression on sm_121 (`posdiv`, main build dev401, no spec, prefix cache off, batch 16384,
    temperature 0, `prompt_logprobs=5`, 8 sequential + 8 concurrent identical requests per prompt, 64-token greedy hashes;
    `notes/data/posdiv/`). The residual divergence after both fixes is NOT noise: it is a deterministic leak from the previous
    forward that exists only with CUDA graphs enabled.** Three arms per prompt (1,460 / 1,999 / 5,960 rendered tokens),
    stock vs `VLLM_MOE_DET_FINALIZE=1` vs det-finalize + `VLLM_QSA_DET_TOPK=1` (kernel-det v2.4, see below):

    | arm | prompt | seq: first div. pos / top-1 flips / mean spread | conc: flips | 64-tok distinct/8 |
    | --- | --- | --- | --- | --- |
    | stock (2 starts) | 1,460 | 1 / 924–988 / 6.4–6.9 | 896–954 | 8 |
    | stock | 1,999 | 1 / 1,042–1,069 / 5.8–5.9 | 1,179–1,258 | 7–8 |
    | stock | 5,960 | 1 / 2,974 / 4.9 | 3,391 | 8 |
    | det-finalize | 1,460 | 1 / 758 / 5.3 | 693 | 4 |
    | det-finalize | 1,999 | 1 / 788 / 4.2 | 1,237 | 2 |
    | det-finalize | 5,960 | 1 / 2,660 / 4.5 | 3,215 | 5 |
    | det-both | 1,460 | 1 / 758 / 5.3 | 693 | 4 |
    | det-both | 1,999 | 1 / 788 / 4.2 | 1,237 | 2 |
    | det-both | 5,960 | 1 / 2,253 / 3.9 | 3,101 | 3 |

    Three readings. (a) Below the 2,048 budget det-both equals det-finalize **to the digit** across two server starts — the
    top-k kernel cannot matter there, and bit-identical statistics rule out any random source. (b) Above the budget the
    deterministic top-k removes a further 15 % of the flips and 2 of 5 distinct completions — ZC502's regression sees the
    #55122 defect. (c) The det arms still "diverge" from position 1 (two tokens of context!) with spreads of whole nats.
    That is not floating point: it is history. `period.py` (16 identical sequential requests, full 1,460-position vector
    hashed): **cold first request = class A, the fifteen after it = class B, bit-identical**; 32-token completions 16/16
    identical; prefix cache on/off makes no difference (prompt-logprob requests bypass it, and a cache-on request that got
    pristine blocks still lands in B, so KV/state *slots* are not the carrier). The eight "sequential" requests of `posdiv`
    disagree only because request 1 followed a different kind of request than 2–8. The sampled-token route shows the same
    thing (`pdiag`: first token 'Let' at −1.19 / −0.03 / −0.54 across three cold-to-warm stock requests), so it is not the
    prompt-logprobs path either.

    Localisation by elimination, each arm a fresh server with det-both and the period test:

    | arm | 16 identical requests | posdiv seq (all 3 prompts) |
    | --- | --- | --- |
    | `--enforce-eager` + per-layer hashes (`lhbis`, 50 modules) | 1 class; 0/50 module hashes differ cold vs warm at 1,460 and 1,999 | — |
    | `--enforce-eager`, no hooks | 1 class | 0 flips, spread 0.000, 1/8 distinct |
    | compile ON, `cudagraph_mode NONE` | 1 class | 0 flips, spread 0.000, 1/8 distinct |
    | compile ON, `cudagraph_mode PIECEWISE` (prod) | 2 classes (A, then B×15) | 758 / 788 / 2,253 flips |
    | compile ON, `FULL_DECODE_ONLY` | 2 classes (A, then B×15) | — |
    | `PIECEWISE` + `--moe-backend emulation` (log-verified) | 2 classes (A, then B×15); 32-tok completions A B C D E then E×11 | — |

    So with the two fixes the forward is bit-exact sequentially in eager and in compiled-without-graphs mode, at every length,
    and every module's output hashes equal — the leak needs cudagraph mode even though a 1,460-token prefill never replays a
    graph (capture sizes 1–8). FULL_DECODE_ONLY leaks the same way, so the trigger is graph capture as such, not the piecewise splitting; the MoE-emulation arm leaks too, so the FlashInfer MoE path is not the carrier, and its 32-token completions *converging* over five requests is the signature of a buffer that fills with traffic. **The carrier is the PLE CPU-offload output buffer** (`plecheck`, `notes/data/posdiv/plecheck.txt`: hash and non-zero
    row count of every PLE layer's GPU output buffer right after each real forward, before the runner releases it). In
    PIECEWISE mode the model consumes the buffer *one step behind*: the first real step (a 32-token warm-up request) saw
    0 non-zero rows, the cold 1,460-token request saw exactly 32 (the previous step's rows), the second and third 1,460
    requests saw all 1,460 rows (the first request's copy, landed late — identical prompt, so "correct" by accident),
    the cold 1,999 request saw exactly 1,460 rows and zeros above. With `cudagraph_mode NONE` every step sees exactly its own rows, and its 1,460-row buffer hash (`101706c574`) and first-token logprob (−0.2638) equal PIECEWISE's *warm* values — so B is the correct computation and the cold A is the broken one: in the served configuration every step runs with the previous step's per-layer embeddings, and only identical consecutive requests hide it. That is class A vs B: the cold request runs with the
    PLE contribution missing for all but a few rows, the warm ones with the previous identical request's rows. The GPU
    side's `vllm::ple_offload_wait` (cuStreamWaitValue32 on the cross-process semaphore) does not hold before the read
    in this mode; a probe inside the wait op (`plewait2`, `pleflag`) shows why: in PIECEWISE every real wait finds the flag already at 1, and the runner-side probe reads 1 at the entry of the very first real step, before any request was submitted, and 1 again after every release — the CPU worker's signal for step k lands after the GPU consumed on the stale 1 and after the release's reset, so the flag is perpetually one step ahead; in NONE mode the first real wait finds 0 and blocks correctly, and later waits see 0 or 1 depending on whether the worker was faster than the launch, with correct rows either way. The unmatched raise originates in engine init, which NONE performs without graph capture. A trace of every semaphore operation in both processes (`plesem`, `notes/data/posdiv/plesem.txt`) names it: `capture_model()` signals dummy outputs (flag 1) and then runs real steps through `execute_model()` — 32, 16, 2, 1 tokens — that submit real requests to the offload worker. The 32-token wait passes on the dummy signal (0 rows), its release resets the flag, the 16-token wait blocks on 0 and is released by the worker's signal *for the 32-token step* (consumes 16 of its 32 rows), and from then on every release is followed within milliseconds by the previous step's late signal. NONE never signals dummy outputs outside `execute_model` and is correct. **Fix:** reset every layer's semaphore on the model stream before a real request is launched (`PleOffloadConnector.prepare_forward`, 11 lines), so the wait can only be satisfied by this step's copy; committed on `jschmied/vllm:ple-offload-wait-fix` on top of the #53899 head. Validation (PIECEWISE, `plefix`): every real step consumes exactly its own rows from the first one (32/16/2/1 at init, then 1,460 ×3, 1,999 ×2; hashes equal to the NONE run's), the cold first request gives the warm logprob (−0.2638), 16 identical requests = 1 class, and the position-resolved set is bit-exact sequentially at 1,460 / 1,999 / 5,960 tokens (0 flips, spread 0.000, 1/8 distinct 64-token completions each); the concurrent batches keep 0 / 416 / 665 flips, identical to the cudagraph-off run — the batch-shape axis, not this defect. What remains after that is the batch-shape axis only: 8 identical prompts in
    one prefill batch diverge from each other (1,460: eager from position 429 / 194 flips, compiled 0; 1,999: from position 1
    in both), the known non-batch-invariance, a separate issue.

    Two traps this run paid for. The FlashInfer autotune cache is keyed on the same `compute_hash()` as the compile cache, so
    the det-finalize arm reused the fused-finalize tactic table and died with `Invalid gemm2 profile id: 59` (per-arm
    `FN_CACHE_ROOT` fixes both caches; see finding 135). And kernel-det v2.3 rejected the main build's block-level indexer
    (`chunk_size 256 smaller than TopK 512`): its host guard applied the cooperative path's sort-buffer constraint to every
    call; v2.4 makes it conditional on `max_seq_len > RADIX_THRESHOLD`, 33 short-row tests added, 210/210 pass
    (`patches/kernel-det/`).

139. **Prod carries the four fixes (2026-09-06 17:4x; `prodcheck`, `notes/data/posdiv/prodcheck.txt`).** `prod_det_overlays.sh`
    installed on `vllm-venv-fnmain2`: deterministic `persistent_topk` (kernel-det v2.4), bit-stable MoE finalize, the FlashInfer
    autotune cache-key backport (needed by the non-fused runner), and the PLE offload semaphore reset (PR #13); the launcher
    defaults the two env-gated ones on (`FN_DET_TOPK=0 FN_DET_FINALIZE=0` = stock arm). Validation start with the documented
    prod configuration (fp8head, 32k context, 16 seqs, batch 16384, **MTP 3, prefix cache on**, PIECEWISE): `QSADET active`
    logged, MoE backend `FLASHINFER_CUTLASS`, 16 identical sequential requests = **one class including the cold first one**
    (full 1,460-position vector hash `17450eec` ×16; 32-token completions 16/16), a tool-call request returns a well-formed
    `tool_calls` finish, speculation live (single-sample acceptance not quoted). Every quality number taken on this build
    between the #53899 port (2026-09-03) and now was measured one PLE step behind and is due for a re-measure.

140. **`FN_MTP` now defaults `disable_eagle_block_drop` on — installed, but the validation start did NOT
    exercise it (2026-09-06 20:46–21:09, `mtpnodrop`, `notes/data/mtpnodrop.txt`).** The launcher
    (`/opt/llm/serve-fnmain.sh`, backup `.pre-mtpnodrop`) builds the MTP shortcut's JSON through
    `FN_SPEC_NODROP:-1`, so the prod recipe now emits
    `{"method":"mtp","num_speculative_tokens":3,"disable_eagle_block_drop":true}`; `FN_SPEC_NODROP=0`
    drops the key again. Parsing verified offline against the fnmain2 venv (`EngineArgs.add_cli_args`
    → `{'method': 'mtp', 'num_speculative_tokens': 3, 'disable_eagle_block_drop': True}` vs `<absent>`).
    **What the two server arms proved and did not prove:** both started, tool calls parsed
    (`finish_reason: tool_calls`), acceptance 210/219 draft tokens, `QSADET active` — but the ON and OFF
    arms returned *byte-identical* counters (219/210 draft/accepted, 319 prefix-cache queries,
    **0 hits in both**), because the probe sent only two identical turns and the first repetition never
    hits (memory `prefix-cache-align-mode-dead`). With zero cache hits the flag has nothing to act on,
    so the arms cannot differ. Two further traps found: the engine's `speculative_config=SpeculativeConfig(...)`
    summary line **truncates before** `disable_eagle_block_drop`, and the launcher `exec`s vLLM so the
    command line never reaches the log — neither is a usable gate. **Open:** a warm-turn probe of ≥4
    identical turns (or `agentloop2.py`, which grows its prefix) × 3 starts, on the re-measurement stack
    of item 3 — finding 141's −26 % is itself inside the one-PLE-step-behind window.

141. **`disable_eagle_block_drop` on the fixed prod stack: −19 % per agent turn, three starts per arm,
    ranges non-overlapping (`mtpnodrop2`, 2026-09-06 21:14–22:28, `notes/data/mtpnodrop2.txt`).**
    Prod recipe (fp8head, 32k, seqs 16, batch 16384, util 0.80, MTP 3, prefix cache on) on
    `vllm-venv-fnmain2` with every fix in place — #55375 stride, PLE semaphore reset, det top-k v2.4,
    det finalize. Probe `agentloop2.py`: 8 turns over a growing ~7.5k prefix, hits from
    `prefix_cache_hits_total` deltas. Arms interleaved ON/OFF/ON/OFF/ON/OFF.

    | arm | s/turn (3 starts) | tokens/loop | acceptance | mean accept len |
    | --- | --- | --- | --- | --- |
    | ON (`FN_SPEC_NODROP=1`, the new default) | **1.76 / 1.74 / 1.75** | 215 | 51.8 % | 2.55 |
    | OFF (`=0`, vLLM's default) | **2.16 / 2.17 / 2.15** | 131 | 42.2 % | 2.27 |

    **The mechanism is visible per turn, and it is what finding 94 predicted.** ON: turn 1 cold
    (`hits+0`, 3.70 s), then `hits+6400` on every turn. OFF: turn 1 **and turn 2** cold (`hits+0`,
    3.63 / 3.40 s), then `hits+4800` — exactly one 1,600-token block less, on every warm turn, plus
    a whole extra cold turn per loop. That hit delta is the gate det-140 lacked: the engine's
    `SpeculativeConfig` log line truncates before the field and the launcher `exec`s vLLM, so neither
    the log nor the command line can confirm the flag — the 4,800 → 6,400 step can.

    **Which number to quote.** The arms take different trajectories (215 against 131 tokens per loop),
    so `ms/tok` (65.3 against 131.8) is **not** like-for-like and must not be quoted as a 2× win. The
    whole-loop figure is **s/turn: 1.750 against 2.160 mean, −19.0 %**. Warm turns only, ON 1.48
    against OFF 1.69 s, is **−12.4 %** — and OFF is slower there while emitting *fewer* tokens per
    turn (11–16 against 22–35), so the fixed-cost gap is wider than that percentage. Finding 94's
    −26 % (2.05 → 1.52 s) is confirmed in direction; ON's warm turn matches (1.48 vs 1.52) while OFF
    has got faster (1.69 vs 2.05), which is what the stride and semaphore fixes should do.

    **Second result, and it may matter more: the MTP restart spread is not reproducing.** All three
    ON starts are identical *to the digit* — 215 tokens, 85 drafts, 255 draft tokens, 132 accepted,
    51.8 %, 2.55 — and so are all three OFF starts; s/turn varies 1.01× within each arm. The 1.83×
    spread that forced the "three starts, report ranges" protocol was measured on the **preview**
    stack (2026-09-01, 66 arms) with the multi-prefill corruption and the PLE semaphore both live.
    This is one workload on one build and does not retire that protocol — it is a reason to test it
    directly in the queued re-measurement (`notes/mtp-remeasure-plan.md`), where cheap stability
    would cut the run substantially.

    Also seen: 268 `NVRM … NV_ERR_NO_MEMORY from _memdescAllocInternal` kernel lines in one burst at
    21:25:59, at an arm transition. No process was killed and no `oom-kill` entry exists; every
    subsequent arm started and completed normally. Read as teardown noise, not a failure.

142. **`top_k_per_row_decode` is faster than our deterministic kernel and is not deterministic — it
    fails the first of the two requirements (`tkprd`, 2026-09-07 06:4x, `notes/data/tkprd.txt`).**
    Asked by gau-nernst on PR #55122 ("deterministic + at least not slower than this PR"). Driven
    through the call convention the model's own AMD QSA path uses (`qwen4_exp/amd/ops/qsa.py:800`,
    where it is the `else` branch to `persistent_topk` on identical inputs), on `vllm-venv-fnmain2`.

    - **Determinism: 0 of 56 shapes.** Six identical calls never reproduce, on every
      rows {1, 8, 64} × n {1k … 40k} × k {512, 2048} × {random, ties} cell, plus all-equal.
    - **Not a buffer artefact.** Re-run with the output pre-filled with two different sentinels
      (−1 and −7): **0 unwritten slots either way**, and the sets agree between sentinels. The
      differences are the kernel's, not uninitialised memory — 9,824 differing slots over five
      pairs at rows=1/n=8k/k=2048, 634,059 at rows=64.
    - **Exactness: same defect class as stock.** Set equals the exact reference on random inputs
      (order does not), but **differs from it on every tie-heavy shape** — precisely #51782 / #54521.
    - **Speed: it is the faster kernel.** Against our det kernel, 0.15–0.92× everywhere. Against
      stock `persistent_topk`, 0.40–0.80× at n ≤ 8k and at 64 rows, but **1.10–1.48× (slower)** at
      16k–32k with 1–8 rows, so it is not a free win against stock either.

    Our own det/stock ratios on this grid are 1.33–4.31×, consistent with the PR body's "1.3–3× per
    call" — which is the number the reviewer read as a "3× regression". The server-level answer is
    already in the thread (finding 82, three starts per arm): **no TTFT and no per-turn cost**.

    **Reading:** the suggestion does not give a shortcut — the fast kernel has the bug the PR exists
    to fix. The constructive direction is the reverse one: the same index-ordered emission and exact
    pivot could be applied to `top_k_per_row_decode`, which is already the non-CUDA branch for this
    model, and would then be both deterministic and faster than what we propose. That is a separate
    PR, not a change to this one. Not offered upstream yet — draft in
    `notes/upstream/comment-55122-tkprd.md`, awaiting the go.

143. **v2.5: the single-CTA path's final sort was a merge all along — worst-case cost ratio 4.31× →
    1.97×, and the shapes this model runs 2.9–3.1× → 1.5× (`kdet25`, 2026-09-07 07:15,
    `notes/data/kdet25.txt`). 210/210 tests pass.** `det_select_row`'s emission writes the
    `> pivot` group at `out[run_gt + rgt]` with `i` ascending under an exclusive prefix sum, and the
    `== pivot` group the same way into its own region — so the row is **two ascending runs** and the
    kernel was running a general bitonic sort (`next_pow2(k)` padded, ~66 sync-separated stages at
    k=2048) over already-sorted data. `det_merge_runs` replaces it: one pass over k, a binary-search
    lower-bound rank in the other run, no inter-stage syncs. Values are row indices, so distinct, so
    the rank is exact and the output is bit-identical to sorting.

    | shape (rows / n / k) | v2.4 det µs | v2.5 det µs | ratio to stock, v2.4 → v2.5 |
    | --- | --- | --- | --- |
    | 1 / 4,096 / 2048 | 26.5 | **11.7** | 4.31× → **1.97×** |
    | 1 / 8,192 / 2048 | 30.4 | **15.3** | 2.95× → **1.48×** |
    | 1 / 16,384 / 2048 | 37.2 | **22.6** | 2.58× → **1.57×** |
    | 64 / 4,096 / 2048 | 49.3 | **20.6** | 3.41× → **1.43×** |
    | 64 / 8,192 / 2048 | 57.5 | **28.8** | 3.10× → **1.51×** |
    | 64 / 16,384 / 2048 | 73.9 | **45.2** | 2.41× → **1.57×** |
    | 64 / 32,768 / 2048 | 126.1 | 125.2 | 2.25× → 2.26× (large path, untouched) |

    **The pattern is the mechanism's signature:** only cells at or below `RADIX_THRESHOLD` (16,384)
    move, and they move most where `k/n` is largest — sort cost scales with k, not n. Every large-path
    cell is unchanged to within noise, which is the control.

    **The large path was deliberately left alone, and the reason matters.** Its `> pivot` emission
    takes output slots with `atomicAdd(&local_histogram[0], 1)` — arrival order, not index order — so
    that region is genuinely unsorted and `det_sort_row` is what makes it deterministic. Merging
    there would have silently reintroduced the ordering bug this whole PR fixes. It looked like a
    symmetric optimisation and was not.

    **Where the remaining cost now sits:** the large path, which holds every ratio above 2× —
    including the new worst cell, 24 rows / 32,768 / k=2048 at 3.72× (and 48 rows at 3.20×, which is
    non-monotonic in row count, so CTA-group packing is involved). Making its gt emission
    index-ordered (BlockScan instead of atomicAdd) would let the merge apply there too. Not attempted.

    Three levers named in the same review and **not** taken yet: the 4 radix passes are unconditional
    (an early exit on `remaining == 0` cannot fire — the bin-selection condition guarantees
    `suf_b1 < remaining`, so remaining never reaches 0; a working exit needs the "threshold bin holds
    exactly `remaining` elements" test instead); the 256-bin suffix scan costs 8 block syncs and runs
    4× where a warp scan would need 1–2; and the emission runs two `BlockScan`s per tile for the
    mutually exclusive `fgt`/`feq` flags where one packed scan would do.

    **Not installed on prod** (`/opt/llm/kernel-det/_C_det.so` is still v2.4) and not pushed to
    PR #55122 — both await the go.

144. **v2.6: three more single-CTA levers — the deterministic kernel now runs at or below stock on
    every shape it owns (`kdet26`, 2026-09-07 07:36, `notes/data/kdet26.txt`). 210/210 pass.**
    All three are output-preserving; none touches the guarantee.

    - **Warp-level bin scan.** The 256-bin suffix sum was 8 double-buffered steps with a
      `__syncthreads()` each, run once per radix pass — **32 block syncs per call**. Now one warp
      does it: lane *l* owns bins [8*l*, 8*l*+8), sums them serially, and a Hillis-Steele suffix scan
      over the 32 lane totals (`__shfl_down_sync`) supplies what lies above each lane. The threshold
      search folds into the same warp, because `suf` of the next lane's first bin *is* that lane's
      `above`. Block syncs here: **zero**.
    - **A working early exit.** The `remaining == 0` test proposed in det-143 provably cannot fire —
      the bin search requires `suf_b1 < remaining`, so the subtraction always leaves ≥ 1. The correct
      test is available once the warp also returns the bin population: when the threshold bin holds
      *exactly* `remaining`, all of it is selected and the lower key bytes cannot change the answer.
      The selection becomes `key >= prefix`, i.e. `key > prefix - 1` with no ties to rank, so
      `prefix -= 1; remaining = 0; break` and the post-loop code is unchanged. Guarded on
      `prefix != 0`.
    - **One packed emission scan.** `fgt`/`feq` are mutually exclusive and a tile holds ≤ N_THREADS
      elements, so both counts fit in 16 bits of one `uint32`: one `ExclusiveSum` and one sync per
      tile instead of two. `static_assert(N_THREADS <= 0xFFFF)` so the packing cannot silently
      overflow if the block size changes.

    | shape (rows / n / k) | v2.4 | v2.5 | **v2.6** |
    | --- | --- | --- | --- |
    | 1 / 4,096 / 2048 | 4.31× | 1.97× | **1.33×** |
    | 1 / 8,192 / 2048 | 2.95× | 1.48× | **1.00×** |
    | 1 / 16,384 / 2048 | 2.58× | 1.57× | **1.14×** |
    | 64 / 1,024 / 512 | 1.66× | 1.17× | **0.71×** |
    | 64 / 4,096 / 2048 | 3.41× | 1.43× | **0.88×** |
    | 64 / 8,192 / 2048 | 3.10× | 1.51× | **1.10×** |
    | 64 / 16,384 / 2048 | 2.41× | 1.57× | **1.07×** |

    Several cells are now **faster than the stock kernel** while being exact and reproducible. The
    worst single-CTA cell is 1.33×.

    **All remaining cost is the large path** (`n > RADIX_THRESHOLD` = 16,384), unchanged across
    v2.4–v2.6 as the control: 24 rows / 32,768 / k=2048 at 3.73×, 48 rows at 3.20×,
    1 / 65,536 / 2048 at 2.32×, 64 / 32,768 / 2048 at 2.18×. v2.7 addresses it.

145. **v2.7: the large path's `gt` emission is index-ordered, so it merges too — worst cell across
    the whole grid 4.31× → 2.45×, 14 of 43 cells at or below stock (`kdet27`, 2026-09-07 07:38,
    `notes/data/kdet27.txt`). 210/210 `test_det.py`, and the PR's own file 134 passed / 26 skipped.**
    The `> pivot` group took its output slots with `atomicAdd(&local_histogram[0], 1)` — thread
    arrival order — which left that region unsorted and made `det_sort_row` load-bearing for
    determinism (det-143 flagged this as the reason not to touch it). Ranking by index with the same
    packed `BlockScan` the eq group uses makes each CTA's slice ascending; CTA *c* covers a lower
    index range than CTA *c+1*, so the region is globally ascending and the merge applies.

    | shape (rows / n / k) | v2.4 | v2.6 | **v2.7** |
    | --- | --- | --- | --- |
    | 1 / 32,768 / 2048 | 1.82× | 1.81× | **1.10×** |
    | 1 / 65,536 / 2048 | 2.36× | 2.32× | **1.49×** |
    | 64 / 32,768 / 2048 | 2.25× | 2.18× | **1.44×** |
    | 64 / 65,536 / 2048 | 2.13× | 2.11× | **1.47×** |
    | 24 / 32,768 / 2048 | 3.72× | 3.73× | **2.45×** |
    | 48 / 32,768 / 2048 | 3.20× | 3.20× | **2.05×** |

    **Whole grid now 1.00–2.45×** (v2.4 was 1.25–4.31×), 14 of 43 cells at or below stock.

    **The two cells still above 2× are a stock-side artefact, not ours.** At n=32,768 / k=2048 our
    kernel takes 55.4 µs at both 24 and 32 rows, while *stock* jumps 22.6 → 39.0 between them. The
    ratio spikes at 24 rows because stock is at a favourable point on its own row-count staircase,
    not because we slow down: our curve is the smoother of the two. Same at 48 rows.

    Cumulative over v2.5–v2.7, nothing given up: same guarantee, same 210 adversarial cases, same
    upstream test file, and the output is bit-identical to the sorted version throughout — every
    change replaces a sort with a merge over data already in order, removes block syncs, or skips
    radix passes that provably cannot change the pivot.

    **Not installed on prod, not pushed to PR #55122** — both await the go.

146. **Re-measured `top_k_per_row_decode` against v2.8: it no longer beats us everywhere, and our own
    posted comparison is now stale in our disfavour (`tkprd2`, 2026-09-07, `notes/data/tkprd2.txt`).**
    det-142 measured it against **v2.4** and we posted "0.15–0.92× of our det kernel" on PR #55122.
    v2.5–v2.8 made our kernel 2–4× faster on the worst shapes, so that ratio is obsolete.

    | shape (rows/n/k) | tkprd / ours, v2.4 (posted) | **tkprd / ours, v2.8** |
    | --- | --- | --- |
    | 1 / 16,384 / 2048 | 0.50 | **1.18 — ours faster** |
    | 1 / 16,384 / 512 | 0.74 | **1.14 — ours faster** |
    | 8 / 16,384 / 2048 | — | **1.13 — ours faster** |
    | 1 / 32,768 / 2048 | 0.66 | **1.08 — ours faster** |
    | 1 / 32,768 / 512 | 0.92 | 1.00 — parity |
    | 64 / 16,384 / 2048 | 0.39 | 0.98 — parity |
    | 1 / 8,192 / 2048 | 0.27 | 0.80 |
    | 64 / 8,192 / 2048 | 0.18 | 0.53 |
    | 64 / 32,768 / 2048 | 0.28 | 0.43 |

    **Range 0.42–1.18× (was 0.15–0.92×).** The split is structural: we are faster on **few rows × long
    rows** (1–8 rows at n ≥ 16k), it is faster on **many rows × short rows** (64 rows at n ≤ 8k, and at
    n=32k/64 rows). The QSA decode shape on this model is few rows — MTP n=3 at c=1 gives 4 query rows —
    so the regime that matters here is the one where we are at parity or ahead.

    Unchanged: it is still **not deterministic on any of the 56 shapes**, and its set still differs from
    the exact reference on every tie-heavy shape. The correctness argument for #55122 is untouched; only
    the speed comparison moved.

    **Owed:** a short correction on PR #55122 — our own comment currently tells a reviewer the
    alternative dominates us, which is no longer true and is an argument against our own PR. Not posted;
    awaiting the go.

147. **Launcher bug: the chunk is sized from the opt-in rather than the dynamic budget, and 8 of 40
    wide-row shapes cannot launch at all on GB10 (found 2026-09-07 while re-sweeping RADIX_THRESHOLD;
    `notes/data/thresh.txt`).** `max_chunk_elements` came from
    `effective_max_smem - kFixedSmemLarge`, but the dynamic limit is the opt-in **minus the kernel's
    static `__shared__`** (4,256 B here), so the chunk is ~1,064 elements too large. Wherever
    `ctas_per_group` resolves to 1 the request overshoots and the launch is **rejected**:
    `dynamic smem 100384 exceeds 97120`, for every row of 24,576 or 49,152 elements at 32 or 64 rows.

    **Not introduced by this week's work** — v2.7 and v3.0 fail on the identical 8 shapes, so it dates
    from the original PR. Our own smem-cap guard (v2.3) is what turns a silent oversubscription into a
    loud error, which is how it surfaced at all. Fixed by querying `cudaFuncGetAttributes` for the
    instantiation that will actually launch and subtracting the static size before sizing the chunk:
    **v3.2 runs all 40 shapes, and row 0 matches the exact reference on every one.**

148. **RADIX_THRESHOLD re-sweep after the merge/scan work: 16384 is still right on GB10, and 32768
    would be worse (`thresh`, 2026-09-07).** Two builds of the same v3.1 kernel differing only in the
    threshold — 131072 (single-CTA throughout the range) and 4096 (cooperative throughout) — timed at
    identical shapes. Median µs, winner and margin:

    | rows | 8k | 12k | 16k | 24k | 32k | 64k |
    | --- | --- | --- | --- | --- | --- | --- |
    | 1 | single +90 % | single +29 % | **single +25 %** | **multi +60 %** | multi +100 % | multi +236 % |
    | 8 | single +66 % | single +33 % | **single +22 %** | **multi +49 %** | multi +94 % | multi +65 % |
    | 64 | single +86 % | single +69 % | **single +47 %** | (blocked by finding 147) | | |

    The crossover sits **between 16,384 and 24,576**, so the current constant is correct and restoring
    upstream's 32768 would cost **60–100 % at n = 24,576–32,768**. I expected the optimisations to have
    pushed the crossover past 32768 and would have argued for reverting the constant on that basis; the
    measurement says the opposite. It does mean the software-barrier exposure that the lower threshold
    creates cannot be reduced by raising it back — that concern stands on its own.

    ⚠️ The 64-row column above 16k is missing because finding 147's launch failure aborted that arm.
    Re-run it on v3.2 before quoting a 64-row crossover.

149. **The single/multi-CTA crossover is row-count dependent on one GPU, so no scalar RADIX_THRESHOLD
    is right everywhere — and v3.1 costs ~5 % on some single-CTA cells (`thresh64` + bisect,
    2026-09-07, `notes/data/thresh64.txt`).** Same two-build method as det-148, now with the smem fix
    so the 64-row arm runs. Winner and margin at k=2048:

    | rows | 8k | 12k | 16k | 24k | 32k | 48k | 64k |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | 1 | S 62 % | S 29 % | S 19 % | **M 61 %** | M 107 % | M 185 % | M 233 % |
    | 8 | S 66 % | S 30 % | S 28 % | **M 72 %** | M 90 % | M 181 % | M 69 % |
    | **32** | S 67 % | S 50 % | S 49 % | **S 35 %** | **S 8 %** | **S 10 %** | **S 15 %** |
    | 64 | S 89 % | S 70 % | S 52 % | M 0 % | **M 11 %** | M 3 % | M 12 % |

    Crossover: **24,576 at 1–8 rows, never at 32 rows, ~32,768 at 64 rows.** `RADIX_THRESHOLD = 16384`
    is therefore correct for 1–8 rows, roughly right at 64, and **wrong at 32 rows, where it sends every
    row ≥ 24,576 to the cooperative path that loses 8–35 %**. This sharpens the portability objection:
    the constant is not merely GPU-dependent, it is row-count dependent on a single GPU, so no scalar
    value is optimal. Making it a function of rows would be another fitted heuristic — the kind #55661
    was closed for — so this is documented, not fixed.

    **Regression found while re-benching, and it is ours.** Bisecting the four builds on
    8 rows / 16,384 / k=2048: v2.7 18.5 µs, v3.0 18.5, **v3.1 19.3–19.5**, v3.2 19.3–19.5, with a
    within-build spread of ±0.1 over three fresh processes — far outside noise. 14 of 43 grid cells
    regress by >0.02 in ratio; the worst absolute cases are 64/4096/2048 (12.4 → 14.3 µs) and several
    n ≤ 16,384 k=2048 cells (+1 µs). **It is not the signed-zero branch** — a build with that branch
    removed measures the same. It entered with v3.1, whose other changes are the `force_single_cta`
    parameter and kernel branch, the active-width geometry (which computes an *identical* chunk for
    these cells), and the CTA-prefix change (multi-CTA only, not executed here). Most likely register
    pressure shifting occupancy under `__launch_bounds__(1024, 2)`; not isolated further.

    Whole grid: v2.7 0.83–2.45×, **v3.2 0.74–2.14×** — better at both ends, with these cells worse.
    ⚠️ The "N of 43 at or below stock" statistic (14 → 9) is not robust: the stock arm is re-timed each
    run and several cells sit within 1 % of 1.00. Quote the range, not the count.

150. **The v3.1 regression is not explained, and the code-layout hypothesis is refuted (2026-09-07).**
    Chasing det-149's ~5 % on single-CTA k=2048 cells, three candidate causes were eliminated by
    experiment rather than argument:

    | hypothesis | test | result |
    | --- | --- | --- |
    | the signed-zero branch in `convert_to_uint32_v2` | build with it removed | **no change** (19.8 vs 19.4 µs) |
    | register pressure / occupancy | `cuobjdump --dump-resource-usage` | **identical**: REG:64, SHARED:5280 on every instantiation, v3.0 and v3.2 |
    | launch geometry for the affected shape | arithmetic | **identical**: `ctas_per_group=1, chunk=16384` either way |
    | code layout — the inlined CTA-prefix block inflating the shared path | `__noinline__` helper | **refuted, and worse**: 20.3 µs against v3.2's 19.3 and v3.0's 18.5 |

    The `__noinline__` split was reverted; the branch keeps the v3.2 kernel. Measurements are three
    fresh processes per build, within-build spread ±0.1 µs.

    **Disposition: disclose, do not keep hunting.** The effect is 0.8–0.9 µs on n ≤ 16,384 at k=2048
    while 18 of 43 cells improve in absolute time and the grid range moves 0.83–2.45× → 0.74–2.14×.
    It was bought for a determinism guarantee that previously had a reachable hole, a launch-rejecting
    smem sizing bug, and an out-of-bounds read on the Filtered path. That trade is defensible in review;
    concealing a measured 5 % is not.


151. **`torch.topk` and the MiniMax-M3 MSA bitonic top-k both fix the determinism bug and both cost
    4.4–35.6× the PR kernel on GB10 (2026-09-07, `notes/data/alt-torchtopk-bitonic.txt`).** Asked by
    gau-nernst on #55122. Harness `/opt/llm/runners/alt_cmp.py`, 27-shape cost grid + 58-shape
    correctness grid, `_C_det.so` from build34 (branch head), 5 × 50 launches.
    - **`torch.topk`: 0 failures on all 58 shapes** — bit-identical over 6 calls, exact set, and after
      an ascending-index sort it equals the index-canonical reference even on the tie-heavy and
      all-equal cases. It would fix the bug. Cost is what rules it out: **4.4–9.9×** with the ragged
      mask and the sort the op's contract needs, and still **1.9–4.7×** stripped bare on a dense row,
      so the gap is the algorithm, not the wrapper. Do not quote the tie behaviour as a contract —
      it is undocumented and this is one PyTorch build on one device.
    - **The bitonic top-k is deterministic and exact by value.** Its 29/58 "set mismatches" against
      our reference are a *tie-choice* difference, not a wrong answer: the selected values equal the
      exact top-k multiset on every case checked, with no duplicates and no out-of-range indices. The
      network breaks ties by position, we break them by lowest index. For reproducibility either is
      sufficient. Cost **5.0–35.6×**, and **8.9–20.8×** at the best cell of a `BLOCK_SIZE_K` ×
      `num_warps` sweep, so it is not a config handicap.
    - **M3's kernel structurally cannot serve k = 2048**: `BLOCK_SIZE_T = next_pow2(topk)` under
      `static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)` needs `BLOCK_SIZE_K ≥ 4096`, and its autotune
      configs stop at 2048. QSA runs k = 2048. Compiling the 4096/8192-wide network by hand takes
      minutes of Triton time per config.
    - Why the gap is that large: both alternatives materialise and order data this op never needs
      ordered. The PR's single-CTA path rescans the row per key byte and writes into final positions.

152. **The Filtered path IS the PR's weak spot, measured on a rented H100 (2026-09-07, 3 starts,
    `notes/data/filtered-H100-run{1,2,3}.txt`).** The one branch GB10 cannot execute
    (`num_rows > 32 && sharedMemPerBlockOptin >= 128 KiB`; GB10 has 99 KiB, H100 227 KiB). Bundle
    `bench/h100-filtered/`, both arms built from source on the box — ours at branch head, upstream at
    the merge-base `d9105ea8` — so the numbers are a ratio, not an absolute. Modal, H100 80GB HBM3,
    sm_90, 132 SMs, torch 2.13.0+cu130.
    - **Correctness: the PR fixes this path too.** det = 0 failures on all 48 shapes in all three
      runs: bit-identical over 6 calls and exactly equal to the reference. Upstream is
      non-reproducible on every shape and loses the SET on every tie-heavy and all-equal case. The
      bug is present on Hopper, not just on GB10.
    - **Cost: 1.01–2.43× over three starts, and it grows with n.** 64 rows: 1.01–1.45× at n=4096,
      1.47–1.85× at 8192, 2.13–2.24× at 20000 (k=512), **2.31–2.43× at 40000** (k=512). The large-n
      cells are stable to ±0.05; only the n=4096 cells move run to run.
    - **Why: the Filtered path is a big win for upstream and a small one for us.** Crossing rows
      32→33 at n=16384, upstream drops 17.5 → 11.4 µs while we drop 23.0 → 18.0, so the ratio opens
      from 1.31× to 1.55–1.58× exactly where the path turns on.
    - **The GB10 grid does not transfer.** Even below the switch (rows ≤ 32, persistent path) H100
      shows 1.31×, against the 0.74–2.14× whole-grid range the PR body quotes from GB10.
    - Affected regime is `rows > 32` — not the c=1 QSA decode shape (4 query rows at MTP n=3), but
      large-batch and prefill. That is the regime H100/A100 deployments actually run.

153. **Fix 1 (device-sized Filtered smem): −23…−30 % at n=40,000, nothing anywhere else — exactly as
    predicted (2026-09-07, 3 starts H100 / 2 SXM4 starts A100, `notes/data/fix1-*.txt`).**
    `FILTERED_TOPK_SMEM_DYNAMIC` was a compile-time 128 KB — the size of the two candidate buffers
    the PR deleted. The smem now caches the row for `det_select_row`, which re-reads GLOBAL memory on
    all four radix passes when the row does not fit, so the useful size is `fixed + n*4`, not a
    constant. 128 KB caches n <= ~32,200; the launcher now queries
    `cudaDevAttrMaxSharedMemoryPerBlockOptin` (cached PER DEVICE), subtracts the per-instantiation
    static smem via `cudaFuncGetAttributes`, and asks for what the widest row needs, floored at the
    old 128 KB.
    - **n=40,000: H100 2.41 → 1.74, A100 2.66 → 2.00** (−23…−30 % on every row count and every k).
    - **n <= 20,000 unchanged** (already cached) and **n=65,536 unchanged** (256 KB of keys exceeds
      every device's opt-in, so it cannot be cached at any request size). Both predicted before the
      run from `det_select_row_bytes`, which is what makes this a confirmed mechanism rather than a
      correlation.
    - Headline barely moves — H100 1.03–2.73 → 1.11–2.72, A100 0.99–3.04 → 1.07–3.04 — because
      n=65,536 sets the maximum. Small-n cells wander a few per cent between runs; the ranges overlap
      and none of it is signal.
    - Correctness: 0 failures on all six post-fix runs, plus GB10's 210-case suite at fails=0 (GB10
      cannot reach this path, so that run proves the change disturbed nothing else).
    - **What is left is fix 2**: after pass 0 only the pivot bin's keys matter (~n/256 on random
      data), so passes 1–3 should be O(n/256), not O(n). That is what upstream's candidate buffers
      did; they were removed because their overflow DROPPED keys. Reintroducing compaction *exactly*
      — sized so it cannot drop, with a deterministic rescan fallback when it would — is the only
      route at n=65,536 and would also cut the 1.0→2.2× climb inside the cached range.

154. **Fix 2 (survivor compaction) WORKS where predicted and is REJECTED anyway — it charges the
    decode shape to pay for long-context prefill (2026-09-07, `notes/data/kdet3[78].txt`,
    `kdet4[0-3].txt`, `fix2-*.txt`; patch kept at `notes/data/fix2-survivor-compaction.patch`).**
    Pass 1 already reads the row and filters to the threshold bin, so it appended that set to a
    shared-memory survivor buffer at no extra traffic; passes 2-3 then read the buffer (~n/256)
    instead of rescanning. Buffer overflow is impossible by construction (`bin_pop` from pass 0 is
    exactly the append count), and an all-equal row falls back to rescanning.
    - **It delivers on the uncached case**: H100 n=65,536 2.67-2.72 → 2.23-2.27, A100 2.92-2.95 →
      2.54-2.60. Compaction demonstrably arms.
    - **It costs the cached case**: +9-15 % on GB10 and at H100 n=20,000. Gating on `!cached` did
      NOT remove the GB10 cost — 3 cells still regress with compaction switched OFF, at identical
      REG:64 / SHARED:5280 (`cuobjdump`), so it is codegen, not traffic or occupancy.
    - **The regressing cells are `rows=1, n=4096/8192` — the decode shape.** The win is at
      `rows>32, n=65,536`, large-batch long-context prefill. Wrong trade; reverted.
    - **Method note, and the reason this was nearly called wrong twice:** a 1-start-vs-3-start
      comparison invented four regressions that a 3-vs-3 comparison did not reproduce. The control
      that settles it is the SAME BINARY run twice — build34 vs build34, 3+3 starts, **0 of 43
      cells** non-overlapping. Only after that control did the remaining differences count as real.
    - **Fix 1 alone: 0 of 43 cells non-overlapping against a 6-start baseline**, and the reverted
      tree reproduces that exactly. Fix 1 ships; fix 2 stays on the patch above, where the next
      thing to try is templating the compaction on a `bool` so the cached path compiles to the
      original code.

155. **ZC502's `vllm-position-parity` collector cannot run Flash-Next on GB10: offline `LLM()` only
    (2026-09-07, `notes/data/vpp*.txt`).** Asked for on #54521 as an sm_121 validation of their
    collector. Code review clean — only subprocess is `git rev-parse HEAD` (arg list, no shell, 2 s
    timeout), no network, no `eval`/`exec`/`pickle`, writes only to `--out`.
    **Three attempts, all stopped by hand before the box hung**, each from a clean start:
    `gpu_memory_utilization` 0.85 → 10 GB free / pressure avg10 34; 0.55 → 0 GB free / 34 GB swap /
    avg10 67; 0.55 after `drop_caches` (119 GB free at start) → 2 GB free / avg10 40 by 15 % of the
    load. Page cache was never the constraint.
    **Cause:** `collect_vllm.py:253` builds its own `LLM(...)`, and a fresh offline engine loads the
    checkpoint through `EngineCore` AND `PleOffloadWorker` concurrently. The PLE half does not scale
    with `gpu_memory_utilization`, so no value of that knob fits it in 128 GB unified. The same model
    serves fine indefinitely as a long-lived server — this is the offline-construction path, not the
    model.
    **Method note:** the harness killed my *shells* on the first attempt for host memory pressure
    while the systemd unit kept running — a background-shell death is a symptom to investigate, not
    the event itself. Check `systemctl is-active` and `free` before concluding anything died.
    **Owed:** if ZC502 adds a server/OpenAI-endpoint mode, run it the same day on the #55122 cases
    (see upstream log 80).

156. **Routing is NOT the fix: the `rows > 32 → FilteredTopK` dispatch is sound (2026-09-07, 1 start,
    `notes/data/routing-ab-H100-persistent.txt`).** Step 1 of the filtered-path plan, and it closes it.
    **Prerequisite, checked first and clean:** there is no row-count limit in the persistent path —
    `num_groups = min(max_resident_ctas / ctas_per_group, num_rows)` is bounded by occupancy, and
    `kDetMaxCtasPerGroup` = 64 caps CTAs per *group*, not rows. `sizeof(RadixRowState)` = 3,588 B
    against vLLM's 1 MiB `RADIX_TOPK_WORKSPACE_SIZE` = room for 292 groups, while `num_groups` can
    never exceed the SM count (132 on H100 → 474 KB). The existing
    `STD_TORCH_CHECK(workspace.numel() >= state_bytes)` would raise rather than corrupt anyway. So
    forcing the routing needed no sizing change — only a bench-only `KDET_NO_FILTERED` env gate in
    `bench/h100-filtered/topk_det.cu` (deliberately NOT in the PR sources).
    - **Persistent wins 3 of 45 cells**, all at 64 rows × 65,536 (77.3 → 53.8 µs, +30 %). It loses
      everywhere else, by up to 99 % at n=20,000 and 70 % at n=40,000.
    - **The killer:** at **128 and 256 rows × 65,536 the persistent path LOSES** (−30 %, −26 %). The
      lone win is an occupancy artifact — at 64 rows the Filtered path's one-CTA-per-row launch uses
      64 of 132 SMs, so the multi-CTA split helps; past the SM count Filtered already fills the
      machine. The window is `num_rows < SMs` AND a very long row, and a dispatch rule for it would
      be SM-count dependent and fragile.
    - The occupancy argument posted to #55122 was therefore **right in mechanism and wrong in scope**;
      say so there rather than let the question stand. The remaining cost is algorithmic → step 3.
    - 1 start only; effect sizes (−99 %, +30 %) are far outside run-to-run spread and consistent
      across all three k, so the direction is safe. Do not quote the percentages without 3 starts.

    **3-start confirmation (2026-09-07 18:1x, `notes/data/routing-persistent-run{1,2,3}.txt`):** every
    cell non-overlapping across 3 v 3 starts. Persistent wins 3 of 45, all at 64x65,536 (+28..32 %);
    loses ~50 % at n=20,000, ~41 % at n=40,000, and — the point that settles it — 23 % at 128x65,536
    and 22 % at 256x65,536. The percentages in this finding are now quotable.
    **Runner note:** the first attempt at these two starts was lost when the harness killed the shell
    holding the loop; the box was at 110 GB free and pressure 0.00 the whole time. Relaunched with
    `setsid` (ppid=1) and it survived. Use `setsid` or systemd-run for anything longer than a turn,
    and never trust a `pgrep` that can match its own command line — that produced two false
    "still running" reports in a row.

157. **CONFIRMED BY CODE, NOT YET BY MEASUREMENT: prod's decode path runs EAGER at c≥4 under MTP
    (2026-09-07, LOW-1 premise check, read-only).** MiaAI's lead, and it holds.
    `CudagraphDispatcher.dispatch()` (`vllm/v1/cudagraph_dispatcher.py`) keys on **`num_tokens`**, not
    on request count, and `num_tokens > max_size` returns `CUDAGraphMode.NONE` — eager, with no
    fallback to a smaller graph. `max_cudagraph_capture_size` is the largest entry of
    `cudagraph_capture_sizes` (`config/compilation.py:695`), which prod sets to `[1,2,4,8]` → 8.
    Under MTP n=3 a verify step submits 1+3 = 4 query tokens per sequence, so c=1 → 4 (captured),
    c=2 → 8 (captured), **c=4 → 16, c=8 → 32, c=16 → 64 — all eager**.
    - **Consequence for a PUBLISHED number:** the Quant Map's "~100 tok/s at c=16 is a bandwidth
      ceiling" was measured on an eager decode path. The ceiling claim is not safe until re-measured
      with capture sizes that cover 4×S.
    - **Explains det-136** (cudagraph A/B null): it ran at c=1, the one width already captured. The
      null was real and also uninformative about c≥4.
    - This is a code-reading result. It predicts that widening `FN_CG_SIZES` to cover 4×S changes c≥4
      throughput and changes nothing at c=1. **Do not quote it as a measurement until that A/B runs**
      — it is exactly the shape of claim that has been wrong before.
    - Owed to a public thread: our MiaAI #19 comment made the c=16 ceiling provisional on this test.

158. **The capture-width A/B is VACUOUS — no cudagraphs were captured in EITHER arm (2026-09-07,
    6 arms, `notes/data/cgsize2.txt`).** Ran det-157's prediction: prod `[1,2,4,8]` vs wide
    `[1,2,4,8,16,32,64]`, 3 starts each, interleaved, c=1/4/16.
    - **Config took effect**: `max_cudagraph_capture_size` reads **8** on all three prod arms and
      **64** on all three wide arms. That check passed.
    - **Result was null everywhere**, control included: c=1 prod 16.6–21.4 vs wide 16.7–21.5 (n=12
      each), c=4 36.3–62.4 vs 36.7–63.8, c=16 46.6–61.3 vs 47.5–65.1. Every range overlaps.
    - **But the null is uninformative**, and this is the point: the wide arm's startup report says
      `0.0 GiB for CUDAGraph memory`, and **no arm logged a single `Capturing CUDA graphs` line**.
      Those are tqdm bars, and the same logs DO contain the checkpoint-loading tqdm bars — so tqdm is
      captured and the absence is real evidence, not a logging artefact. **Neither arm built graphs.**
      The experiment compared no-graphs to no-graphs.
    - **The interesting question is now different**: does prod capture cudagraphs *at all*? If not,
      det-136's null cudagraph A/B has a second possible explanation, and det-157's dispatcher
      reading — correct as source analysis, and confirmed by `max_cudagraph_capture_size: 8` in the
      live log — describes a path that may be moot because nothing is captured anyway.
    - **Decisive next test, queued, cheap:** add a third arm at `FN_CG_MODE=NONE`. If NONE lands in
      the same range as PIECEWISE and wide, cudagraphs are inert in this configuration entirely and
      the whole line of inquiry resolves differently. Do NOT re-run the width sweep before that.
    - Start 3 of both arms ran during heavy host filesystem IO (my own disk audit). prod3 measured
      56.1/48.9 at c=16, inside the prod1/prod2 range, so it is not an outlier — but the asymmetry is
      recorded rather than hidden.
    - **Owed:** the MiaAI #19 commitment ("our c=16 numbers are provisional until this is tested") is
      NOT discharged by this run. Nothing to post until the NONE arm settles what is actually running.

159. **Our prod has been serving dense NVFP4 linears through a W4A16 kernel, not W4A4 (2026-09-07,
    verified in the prod venv; upstream #55397, fix #55405).** `_POSSIBLE_NVFP4_KERNELS[CUDA]` is
    scanned first-match-wins. Position 1 (`FlashInferCuteDslNvFp4LinearKernel`, W4A4) is gated to
    sm_10x and rejects GB10; position 2 (`FlashInferCuteDslNvFp4W4A16LinearKernel`) has the gate
    `cc not in (100,103) and not (120 <= cc < 130)` → **sm_121 passes**, so the scan stops there and
    the three native W4A4 kernels below (FlashInferCutlass, B12x, Cutlass) are never reached.
    Confirmed by calling `is_supported(121)` down the list in `vllm-venv-fnmain2`, not inferred from
    the source: first match is the W4A16 kernel.
    - **Scope:** dense/attention projections, not the MoE experts (those take the fused MoE path,
      `moe_backend='auto'` → `fused_moe`). Flash-Next carries substantial dense weight, so it is not
      a small surface — [[flashnext-single-stream-limit-and-mtp]] put 69 % of single-stream in BF16
      GEMV on unquantized dense weights.
    - **Effect upstream calls it:** a prefill regression on GB10. Our TTFT numbers are measured
      through this path.
    - **HYPOTHESIS, not yet tested — this could confound [[w4a16-vs-w4a4-measured]].** That finding
      compared two *checkpoints* (W4A16 vs W4A4 quantization schemes) and concluded W4A4 stays
      because W4A16 buys only 0.42 pp of BF16 fidelity for 15 % decode and +8.7 s TTFT. If the
      "W4A4" checkpoint's dense linears were being served by a 16-bit-activation kernel anyway, the
      fidelity gap would be compressed toward zero — which is roughly what we measured. **Do not
      restate that finding until this is checked**; equally, do not withdraw it, because the MoE
      path (which dominates the parameter count) is unaffected.
    - **Action, queued behind `mtprem`:** apply #55405 (reorder the list so W4A16 ranks below the
      native W4A4 kernels), confirm `is_supported` selection flips, then A/B prefill/TTFT and the
      c=1 decode ladder. We have the affected hardware and the issue author does not appear to —
      this is a cheap, high-value contribution to a fix that is already written.

160. **ngram / ngram_gpu speculation is IMPOSSIBLE on this stack: it forces model runner V1, and
    `VLLM_PLE_CPU_OFFLOAD` refuses V1 (2026-09-07, `mtprem` group 1).** Both ngram arms died at engine
    init after 50 s with:
    `Model Runner V2 does not yet support ngram/ngram_gpu speculative decoding; using the V1 model
    runner instead` → `ValueError: VLLM_PLE_CPU_OFFLOAD does not support the requested configuration.
    Unsupported settings: model runner V1`.
    - **Not a misconfiguration.** PLE offload is not optional for Flash-Next on a 128 GB GB10 (the PLE
      tables are ~51 GB), so ngram and this model cannot coexist on the main build.
    - **Consequence for the published page:** the Quant Map's ngram / ngram_gpu comparison **cannot be
      re-measured on the current stack at all**. It was measured on the preview build
      (`vllm-venv-fnext`), where this restriction evidently did not apply. Those cells are therefore
      not merely stale — they are unreproducible without either a V1-capable PLE path or a build that
      does not need the offload. State that on the page rather than silently dropping the rows.
    - `mtprem` group 1 therefore runs 3 of 5 arms (nospec / MTP k=2 / MTP k=3), which are the arms the
      headline claim rests on; 6 starts are saved and the run finishes sooner.
    - Untested alternative if the comparison is ever wanted: `FN_PLE_OFFLOAD=0` puts the tables on the
      GPU. Whether they fit alongside a 0.80 utilisation KV pool is unknown, and it would no longer be
      the same configuration as every other arm — so it is a different measurement, not a repair.

161. **A production Flash-Next NVFP4 deployment on GB10 exists with a different PLE path and no
    cudagraphs (2026-09-07, [Radar105/qwen38-flash-next-nvfp4-spark](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark)).**
    Same model, same NVFP4 weights, same class of box, 262K context, three hours a day of agentic use.
    - **`VLLM_QWEN4_PLE_MMAP=1` instead of `VLLM_PLE_CPU_OFFLOAD`.** Checked: that variable exists in
      neither our venv nor upstream main, so it is theirs, carried on a newer base (`7fbd44cb`) than
      our merge-base. If it avoids the V1 rejection at `uniproc_executor.py:71`, it repairs what
      det-160 declared impossible — the ngram/ngram_gpu arms — and gives us a second PLE
      implementation to compare against the offload worker's 16 page faults per token.
    - **`--enforce-eager` in production.** Independent support for det-158: an operator running this
      model daily turned cudagraphs off. Our own capture-width A/B was void because no graphs were
      captured in either arm; this suggests that may be the normal state rather than a misconfiguration.
    - **Their patch stack is our determinism chain**: #55375 (merged), #53798 and #54076 (the two legs
      still open, both of which we have read this week), plus #54713 and #55390, which are new to us.
      **They do not carry #55122**, so their production retains the QSA top-k nondeterminism — a
      concrete answer to "who else needs this fix".
    - **Cross-check on our numbers, and they agree**: 26.83 tok/s decode at 47,643 input, 33.84 at
      29,985, production median 22.7 / peak 35.1 over 392 decode windows. Our c=1 cells the same night
      are 26.7–28.0 at short prompts. No discrepancy to chase.
    - Different operating point from ours in ways that matter when comparing: `--max-num-seqs 1`,
      `--kv-cache-memory-bytes 8G` (explicit rather than a utilisation fraction),
      `--mamba-cache-mode align --prefix-cache-retention-interval 1600`, MTP **2** not 3.

162. **GROUP 0a: the 1.83× MTP restart spread is GONE — 1.15× on the fixed stack (2026-09-07,
    3 starts × 3 reps, `mtprem` group 0a).** The plan's whole "three starts, report ranges" protocol
    was written against a 1.83× within-config spread (MTP2, 47.8 → 77.5 ms/tok) measured on the
    **preview** stack with the multi-prefill corruption and the PLE semaphore both live.
    | start | c=1 tok/s | acceptance |
    | --- | --- | --- |
    | 1 | 26.7 / 28.0 / 27.9 | 60.7–67.5 % |
    | 2 | 24.7 / 27.8 / 25.8 | 60.7–71.5 % |
    | 3 | 26.1 / 28.3 / 27.1 | 60.1–65.9 % |
    **Across all nine: 24.7–28.3 tok/s, spread 1.15×** — and the within-start spread (1.05–1.13×) is
    most of it, so restart-to-restart adds almost nothing. That is the no-spec/ngram regime (1.09–1.10×),
    not the old MTP regime.
    - **Consequence for the plan:** three starts per MTP cell was a defence against a defect that has
      since been fixed. Later groups could drop to two starts, roughly halving the remaining run.
      **Not changing `mtprem` mid-flight** — it is already running groups 1–3 at three starts, and
      re-cutting the protocol while the job is in progress would make the groups incomparable to each
      other. Apply it to the *next* MTP run, not this one.
    - This also retires [[mtp-restart-instability]] as a live constraint: it was real, it was caused by
      the corruption and the semaphore, and both are fixed in `vllm-venv-fnmain2`.
    - Caveat: one workload (short agent-style prompt, 551 tokens, c=1). The old 1.83× was measured at a
      different cell. A single workload closing the spread is strong evidence, not proof, that it
      closed everywhere.

163. **MTP RE-MEASUREMENT COMPLETE on the fixed prod stack (2026-09-07/08, `mtprem`, 69 cells,
    `notes/data/mtprem.txt`).** Groups 0a/1/2/3, three starts each, arms interleaved across starts,
    per-arm `FN_CACHE_ROOT`, `vllm-venv-fnmain2` with the multi-prefill fix (#55375), the PLE
    semaphore reset and `disable_eagle_block_drop` all in place. FN_UTIL 0.80, 8k context, batch 4096,
    max-num-seqs 16, prompt ~550 tokens.

    | group | cell | n | tok/s | acceptance | accept-len |
    | --- | --- | --- | --- | --- | --- |
    | 1 | c=1 no-spec | 6 | **15.1–15.9** | — | — |
    | 1 | c=1 MTP k=2 | 6 | **25.7–28.8** | 68–81 % | 2.35–2.61 |
    | 1 | c=1 MTP k=3 | 6 | **26.4–27.5** | 61–70 % | 2.82–3.10 |
    | 2 | c=16 off | 6 | **115.5–130.6** | — | — |
    | 2 | c=16 MTP k=2 | 6 | **154.1–169.8** | 73–75 % | 2.46–2.49 |
    | 2 | c=16 MTP k=3 | 6 | **156.8–164.4** | 65–67 % | 2.94–3.01 |
    | 3 | c=1 n=2 | 6 | 25.7–28.0 | 71–80 % | 2.42–2.59 |
    | 3 | c=1 n=3 | 6 | 25.0–28.5 | 62–70 % | 2.84–3.10 |
    | 3 | c=1 n=4 | 6 | 24.1–29.9 | 51–64 % | 3.02–3.56 |
    | 3 | c=1 n=9 | 6 | **18.7–24.6** | 28–38 % | 3.56–4.45 |

    **What is now established.**
    - **MTP over no-spec at c=1 is 1.62–1.91× (k=2) and 1.66–1.82× (k=3)** — both arms cleanly
      non-overlapping with no-spec. The page's headline claim survives, on a stack that is not
      corrupting its own output.
    - **At c=16 the gain shrinks to 1.18–1.47×.** Batching already fills the machine, so speculation
      has less idle to reclaim. Health checks pass: accept-length sits below its ceiling rather than
      pinned at it (the corruption signature), zero preemptions, per-request times clustered within
      0.4 s across all sixteen streams.
    - **The published c=16 cells (99.1 / 100.5 tok/s) are BELOW even our no-spec arm (115.5–130.6).**
      Those cells were measured while the multi-prefill corruption was live and are understated, not
      merely stale.
    - **Depth: flat through n=2–4, then falls.** Acceptance rate decays monotonically (80 % → 28 %)
      while accepted length rises (2.4 → 4.5); the two cancel to n=4 and the rate collapse wins by
      n=9, costing ~25 %. n=2/3/4 overlap each other completely, so the honest statement is "flat to
      4", not an ordering. **n=5–8 remain unmeasurable** (the `block_size`/`compress_ratio` hole), so
      the curve has a permanent gap between 4 and 9.
    - **k=2 vs k=3 is unresolved at both concurrencies** — the ranges overlap. What differs is the
      mechanism, consistently: k=3 buys ~0.45 more accepted tokens per draft at ~8 pp lower acceptance.

    **What could NOT be measured, and why it matters for the page.**
    - **ngram / ngram_gpu: impossible on this stack** (det-160) — they force model runner V1 and
      `VLLM_PLE_CPU_OFFLOAD` rejects V1. 6 of the run's 6 deaths are these arms, 2 per start. Those
      page cells are unreproducible, not stale.
    - Group 4 (the ~68-token threshold) and group 5 (FP8-KV c=1) were left out of this run by choice —
      lowest value in the plan, and I preferred the night to finish clean.
    - Every figure here is the **main** build (`0.28.1rc1.dev401`). The page was measured on the
      **preview** build, so the non-MTP ladder on the page is still owed a re-run for the same reason.

164. **NVFP4 kernel selection is a NULL on Flash-Next — the checkpoint excludes the layers the bug
    affects (2026-09-08, 6 arms, 3 starts each, `notes/data/nvfp4k.txt`).** Tested det-159 with one
    variable: `VLLM_DISABLED_KERNELS=FlashInferCuteDslNvFp4W4A16LinearKernel`.
    | metric | stock (W4A16) | w4a4 (native) | |
    | --- | --- | --- | --- |
    | cold prefill @8k | 2.90–2.93 s | 2.91–2.93 s | overlap, 1.002× |
    | cold prefill @30k | 8.18–8.24 s | 8.16–8.28 s | overlap, 1.000× |
    | decode c=4 | 46.3–52.7 tok/s | 50.4–54.1 | overlap |
    - **Mechanism verified independently of the run**, because vLLM does not log the chosen kernel
      class at default verbosity and the harness's `kernel=` field came back empty. Calling the
      selection with and without the env var: unset → `FlashInferCuteDslNvFp4W4A16LinearKernel`,
      set → `FlashInferCutlassNvFp4LinearKernel`. So the arms genuinely differ; the empty log field
      is a logging gap, not evidence of a vacuous A/B (contrast det-158, where nothing differed).
    - **Why null here while upstream measured −31.6 %:** the checkpoint's `exclude_modules` covers
      `*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`, `*.mlp.shared_expert.*` — the dense and
      attention projections this kernel serves are **BF16 on this model**, and the NVFP4 weights are
      in the MoE experts, which take the fused MoE path and never reach
      `_POSSIBLE_NVFP4_KERNELS`. #55397 measured a **dense 27B**, where those projections are NVFP4.
      The bug is real; there is almost nothing on Flash-Next for it to be wrong about.
    - **This substantially weakens the confound flagged in det-159.** Our W4A16-vs-W4A4 fidelity
      comparison ran through the MoE path, which this selection never touches. Downgrade
      [[w4a16-vs-w4a4-measured]] from "suspect" back to "stands", with the scope now measured rather
      than assumed.
    - **Measurement trap, recorded because it nearly produced the right answer for the wrong reason:**
      the TTFT probe prints `median=0.60s` while `all=2.93 0.60 0.60`. With prefix caching on, reps
      2–3 hit the cache, so the median reports WARM time and would have shown "no effect" regardless
      of the kernel. Only the first request is a prefill measurement.

165. **#55314 measured on sm_121: it fixes the SET essentially for free, and does NOT fix the order —
    which makes the union worth building (2026-09-08, `notes/data/tkunion.txt`).** First independent
    run of that PR on this hardware; their kernel built standalone against their own base's launcher
    (`bench/topk-union/their`).

    | property, 81 shapes | stock | **#55314** | #55122 (ours) |
    | --- | --- | --- | --- |
    | self-consistent over 6 calls | 0 | **0** | **81** |
    | valid exact top-k by value (16 tie-heavy shapes) | — | **16 / 16** | 16 / 16 |
    | index-canonical order | 0 | 0 | **81** |
    | cost vs stock | 1.00× | **1.00–1.15×** | 0.88–1.54× |

    - **Their exactness fix works and is nearly free.** On the tie-heavy and all-equal shapes their
      selection is always a valid exact top-k by value: 0 of 16 wrong. Cost 1.00–1.15× of stock.
    - **It does not touch reproducibility.** 81 of 81 shapes still fail self-consistency over six
      calls, exactly as the retained `atomicAdd` slot assignment predicts.
    - **Ours costs most where theirs costs nothing**: 1.24× at 64×16384, **1.54× at 64×32768**, while
      theirs is 1.01–1.04× at the same cells. That is the price of removing the candidate buffers.
    - **NEAR-ERROR, recorded because it would have been a public accusation.** My first pass scored
      "set" against the *index-canonical* reference (lowest-index ties) and reported #55314 failing 52
      of 81 — I was one step from posting that their fix does not work. Their claim is matching
      `torch.topk`, which does not fix tie identity. Re-scored on the VALUE multiset, they pass
      everything. Same trap as det-151 with the bitonic kernel: **a "wrong set" against a canonical
      reference is usually a different valid tie choice, not a wrong answer.** Check values before
      claiming anyone's kernel is broken.
    - **Conclusion for the union:** their selection is the cheap half and our emission is the correct
      half, and the numbers now say so rather than the argument. Their approach + `union_emit_ordered`
      should land near 1.0–1.2× of stock *with* reproducibility, against our current 1.54× worst cell.
      That is worth building and is a better artefact than either PR alone.

166. **The union is WORSE than our own PR as a post-pass — do not propose it (2026-09-08,
    `notes/data/tkunion2.txt`).** Built #55314's selection + #55122's ordering and measured it.
    Design: a post-pass that never touches their descent, so it cannot break their set — read the
    pivot back off their own output (min ordered key among the indices they selected), count elements
    above it, then emit `> pivot` plus the lowest-index ties in one ordered pass.
    - **It works.** Wiring it into `histogram_256_topk` made exactly the shapes that use that path
      (8192 < n ≤ 32768, i.e. n=20000 in the grid) fully self-consistent — 18/81 shapes, and *only*
      that path — while the value check stayed 0/16 wrong. Path-by-path determinism, as designed.
    - **And it costs too much.** On the wired path it is **1.76–1.99× stock** against our #55122 at
      **1.14–1.31×** on the same cells. The safety that makes it unable to break their set — running
      after them rather than inside them — is what makes it expensive: two extra full-row scans, one
      to count above-pivot and one to emit. That is roughly +0.8× of stock, and it swamps the
      1.00–1.15× their selection costs.
    - **Conclusion: our PR is better than the obvious combination.** The "their set + our order" idea
      is right in principle and wrong as a bolt-on. Making it pay would require *fusing* the ordered
      emission into their descent instead of appending it — which is what I avoided because their
      coarse bucket is fp16-derived (`convert_to_uint8` → `__float2half_rn`) and is not byte 3 of the
      fp32 key, so the pivot cannot be assembled naively from their internals. That is a real piece of
      work with an uncertain payoff.
    - **Nothing proposed upstream.** The comment I had drafted offering to write the ordering patch
      against their branch is withdrawn on these numbers. What I can honestly say on #55314 is what
      det-165 already established: their exactness fix works and is nearly free, and it does not
      address reproducibility — no offer attached.
    - Two paths remain unwired (`histogram_2048_topk` for n ≤ 8192, `radix_topk` for n > 32768); the
      latter is multi-CTA and a post-pass there would need cross-CTA coordination. Not worth doing
      given the cost result.

167. **FOUR INDEPENDENT DESIGNS FOR A MERGED TOP-K, SYNTHESISED (2026-09-08, three subagents +
    the user's own opinion; full texts in `notes/data/topk-merge-opinion-{A,B,C}.md`).** Three agents
    got the same brief — read stock, #55314 and our #55122 on the box, propose a merged kernel that
    keeps #55314's selection cost and our ordering guarantee — with no hint of any design of mine.
    A fourth opinion came from the user afterwards. What follows is where they agree, where they
    disagree, and what I checked myself.

    **Unanimous, and each agent reached it independently:**
    - **The 1.54× cell at 64×32768 is a routing artifact, not the cost of determinism.** Our PR
      lowered `RADIX_THRESHOLD` 32768 → 16384 because `det_select_row` caches a 4-byte key per element
      and 32768×4 = 128 KB does not fit in GB10's 101376 B optin. The dispatch is
      `seq_len <= RADIX_THRESHOLD` (`persistent_topk.cuh:1190`), so at n=32768 stock runs **one CTA**
      through `histogram_256_topk` while we run a **2-CTA cooperative launch**. I verified both
      constants and the comparison operator myself. That cell compares two different algorithms.
    - **#55314 is not a merge candidate on its own.** 0/81 self-consistent, and its two tie clips
      (`atomicAdd(&shared_final_k,-1)` and the `bp < DBUF` stash clip, reachable at `level == 4`) make
      the *set* scheduling-dependent, not just the order. It cannot be rescued by a cheap order fix.
      This restates det-165/166 and all three agents arrived at it from the source.
    - **Blocked emission, 4 items per thread.** Our emission runs one `cub::BlockScan` +
      `__syncthreads()` per 1024 elements — 16 scans at n=16384, 32 at 32768. Thread `t` owning
      `[4t,4t+4)` cuts that 4×, and `pos = g + min(e, fin)` is unchanged because it is a pure function
      of index, pivot and `fin`. All three agents proposed it; so did the user. **Highest
      value/risk ratio change in the whole set** — it cannot alter the selected set or its order.
    - **Nobody has measured what share of a decode step this kernel is.** All three name it as the
      gate that could end the project: at ≤3 % of the step, 1.30× costs <1 % end to end and the
      correct action is to merge #55122 unchanged and spend the effort elsewhere.
    - **Merge #55122 now, optimise in a follow-up PR.** Unanimous.

    **A real bug all three found, in stock — not in us.** `decode_bin` and `convert_to_uint8` do not
    canonicalise signed zero, so `-0.0f` and `+0.0f` land in different coarse bins and the fp16 bucket
    stops being monotone in the fp32 order exactly at the pivot. Our `convert_to_uint32_v2` fixes it
    (line 53); #55314 does not fix it anywhere. **On our branch both functions are dead code**
    (agent B confirmed by grep: nothing calls them), so this is not a defect in our shipped path —
    it is a defect in stock and in #55314, and it belongs in a note to their author, not in our PR.

    **Where they disagree, and who is right:**
    - **B's diagnosis of the 1.54× cell is built on a false premise.** B asserts "at 64×32768 both
      stock and #55122 do 1 global + 7 shared per chunk — identical pass counts" and concludes the
      regression must be the `BlockScan` barriers and the inter-CTA round trip. But stock's threshold
      is 32768 and the test is `<=`, so stock is single-CTA at that width. A and C have it right.
      B's barrier argument may still be a real second-order cost; its stated evidence is not.
    - **The coarse first bucket.** A and C would reuse stock's 11-bit fp16-derived `decode_bin` inside
      `det_select_row`. B rejects it: the fp16 bucket is monotone but not injective at the extremes —
      everything under ~6e-5 collapses to the zero bin and everything over 65504 to the inf bin, which
      is precisely what forces #55314's descent — and proposes 4096 bins on `key >> 20` taken straight
      out of the ordered fp32 key, which has no such collapse and no fp16 round trip. **B is right
      here**, and it matters because a finer, non-collapsing first split is what makes the candidate
      set small enough for the later passes to be cheap.
    - **Bitmap vs candidate stash.** A (§2.2) and C (§2) both converge on the user's selected-index
      bitset: two shared bitmaps (`> pivot`, `== pivot`), written only by `atomicOr` (commutative and
      idempotent, so order-independent), then one packed prefix scan over `n/32` words to produce
      every output slot. That is the same `pos = g + min(e, fin)` formula evaluated once per 32
      elements, and it is the only design here that also removes stash-capacity clipping as a
      correctness concern. B instead keeps a candidate stash but makes its capacity unable to affect
      the result (overflow falls back to a full shared scan; the stash only ever feeds commutative
      histogram adds). Both are sound; the bitmap is the smaller correctness argument.
    - **C's trap that nobody else named:** for any bitmap design on the multi-CTA path,
      `chunk_size` must be a multiple of 32, or two CTAs race on one bitmap word. A host-side
      `STD_TORCH_CHECK` turns that from a data race into a launch error.
    - **A's test that nobody else named, and the cheapest in the set:** the whole family rests on the
      coarse bucket being monotone in the fp32 order. Verify it host-only and exhaustively — all
      65536 fp16 patterns, assert the rounding intervals are ordered. Seconds, no GPU, and it pins
      the invariant against a future "optimisation" of `decode_bin`.
    - **B's unique contribution, and it matches the user's fourth opinion:** the multi-CTA `gt`/`eq`
      counts need not be obtained by scanning the chunk twice — in each round, after `thr_r` is known
      and before `local_histogram` is zeroed, accumulate `Σ_{b > thr_r} local_histogram[b]`. That
      deletes two full shared chunk passes for four 256-element warp reductions.

    **What this changes about the plan.** The merged kernel is real and both bitmap variants would
    work, but every one of the four opinions says *not to build it yet*, and for the same reason:
    the two cheapest changes attack the same costs with none of the architectural risk, in a file two
    PRs are already contending over. Order of work:
    1. `RADIX_THRESHOLD` 16384 → 20480 (`thr`, queued). `det_select_row` caches while
       `fixed + 4n <= smem`; static 4256 B → dyn 97120 B → caching holds to n = 22712, so 20480 is
       legal with margin. If it moves the regressed band back onto the cached path, the honest fix to
       the PR is a better threshold plus a corrected sentence, not a caveat.
    2. Blocked 4-item emission. ~60 lines, provably set- and order-preserving.
    3. The multi-CTA counts from the round histograms (B's L3b).
    4. Only then, against the post-(1..3) numbers, decide whether the bitmap is worth building.
    And before any of it, the number that could end the project: this kernel's share of a decode step.

    **Caveat on the exercise itself.** My brief to the agents contained two errors, both caught by
    agent B: I wrote that #55122 "rescans the row" when `det_select_row` caches the ordered keys in
    shared memory whenever they fit (always, on GB10's single-CTA domain), and I wrote "three code
    paths" when `histogram_2048_topk` and `histogram_256_topk` are dead on our branch. Neither error
    propagated into a wrong recommendation, but the traffic-budget framing I gave them was wrong and
    B is the only one that said so.

168. **CORRECTION TO det-167: the "gate nobody measured" was measured, twice, and it closes the
    optimisation question (2026-09-08).** All four opinions in det-167 converged on the same
    caveat — *what fraction of a decode step is `persistent_topk`? nobody has produced that number,
    and it decides everything.* They were wrong, and so was I for repeating it: the number exists,
    from two independent parties, and it has been sitting in the PR thread since 03/04 Sep.

    | source | measurement | result |
    | --- | --- | --- |
    | us, 2026-09-04 (PR #55122 comment) | TTFT at 7.5k and 29k, 8-turn agent loop, 3 server starts per arm, MTP n=5 | **every pair inside the start-to-start band** — no measurable TTFT or per-turn cost |
    | k3dani, 2026-09-03 (independent, different config: MTP=2, PIECEWISE, 8k chunks, FlashInfer 0.6.17) | prefill throughput at 6,082 and 24,416 tokens | **99–100 % of stock** (the 87–90 % figure in that comment is the *Python `torch.topk` fallback*, not this kernel) |

    So the kernel's 1.14–1.30× microbenchmark ratio is **≤1 % end to end**, on two different
    configurations, measured by two parties who did not coordinate. Consequences:

    - **The merged bitmap kernel is closed, not deferred.** Every version of it — A's, C's, the
      user's — exists to convert a ≤1 % end-to-end cost into a smaller ≤1 %. It touches four or five
      kernels and two shared-memory layouts to do it. There is no throughput argument left.
    - **The cheap changes survive, but their justification changes.** `RADIX_THRESHOLD` (`thr`,
      queued) and the blocked 4-item emission (`blkem`, queued, branch
      `perf/topk-blocked-emission`) are now **review hygiene, not performance work**: the
      microbenchmark table is what a reviewer sees, and a 1.54× cell invites an objection that costs
      more review time than the fix costs to make. Both are also cheap enough that this reframing
      does not change whether to run them — only what I would claim if they win.
    - **B's L3b and the ±0 fix are unaffected** — the former is a code simplification (two chunk
      passes deleted), the latter is a correctness bug in stock and #55314.

    **Process lesson — corrected, and it is the actual finding here.** My first version of this
    entry blamed the notes: results posted upstream but never folded back. **That was wrong.** Both
    numbers are recorded, in this file, as **finding 76** (k3dani's validation, including the
    99–100 % prefill line and the explicit note that 87–90 % is the Python fallback) and **finding
    82** (`kdetab`, the three-start server A/B, every pair inside the start-to-start band). They
    were written the day they were measured and they are in the same file I appended det-167 to.

    So the failure is **retrieval, not recording**. This file is 2,400+ lines of chronological
    findings with no index by question. Three agents reading the repo missed findings 76 and 82, and
    so did I while writing a synthesis into the file that contains them — and then I wrote a process
    lesson blaming the recording, which a two-minute grep would have refuted. A chronological log
    answers "what happened on the 4th" and cannot answer "what does this kernel cost end to end",
    which is the question anyone actually arrives with.

    Remedy, and it is cheap: a question-indexed header at the top of this file — *is the kernel
    deterministic / what does it cost per call / what does it cost end to end / does it change
    quality* — each pointing at the finding numbers that settle it. Written below as part of this
    entry. The knowledge tree exists for exactly this and this file is not in it.

169. **PIECEWISE vs NONE cudagraphs: NULL where the graph is reachable, VACUOUS where it is not
    (2026-09-08, `cgnone2`, 6 arms interleaved, ~19 min each, raw `notes/data/cgnone2.txt`).**
    Prod capture sizes `[1,2,4,8]`, `FN_MTP=3`, `FN_SEQS=16`, c ∈ {1,4,16}. **Mechanism check passed
    on every arm** — PIECEWISE arms log `max_cudagraph_capture_size: 8`, NONE arms log `0` — so
    unlike `cgsize2` this run is not vacuous by construction.

    Compared **like rep against like rep**, never a mean over reps (reps inside a start are not
    exchangeable — see `notes/data/cgnone2-contamination.md`), tok/s aggregate, 3 arms per mode:

    | c | rep | PIECEWISE | NONE | verdict |
    | --- | --- | --- | --- | --- |
    | 1 | 0 | 18.2 / 18.8 / 19.0 | 18.7 / 18.8 / 19.1 | overlap |
    | 1 | 1 | 16.2 / 16.6 / 16.8 | 16.4 / 16.8 / 16.8 | overlap |
    | 1 | 2 | 19.7 / 19.8 / 19.9 | 19.2 / 19.7 / 19.9 | overlap |
    | 1 | 3 | 21.0 / 21.4 / 21.6 | 21.0 / 21.1 / 21.4 | overlap |
    | 4 | 0–2 | 36.9–64.1 | 34.7–63.2 | overlap, **but vacuous** |
    | 16 | 0–1 | 48.2–59.4 | 46.7–61.7 | overlap, **but vacuous** |

    **0 of 9 cells non-overlapping — no effect.** Acceptance is identical to the digit across modes
    (38.4–61.0 % in both, n=27 each), as it must be: cudagraph mode cannot change what is sampled.

    **Which cells actually mean anything.** MTP n=3 issues 4 query tokens per sequence, so the decode
    batch is 4·c tokens against a captured maximum of 8:
    - **c=1 → 4 tokens, inside the capture set: the graph IS used.** This is the one informative
      cell, and it is a real null — *turning cudagraphs off entirely costs nothing measurable at
      c=1*, across 3 arms × 4 reps.
    - **c=4 → 16 tokens and c=16 → 64 tokens, both above 8: BOTH arms run eager.** Their overlap is
      not evidence about cudagraphs; it is two eager arms agreeing with each other. Same vacuity trap
      as `cgsize2`, caught this time before it was written up as a result.

    **This does NOT discharge the MiaAI-Lab #19 commitment, and I had expected it to.** We told them
    our c=16 numbers were provisional because prod's capture sizes may leave c≥4 eager. This run
    *confirms the arithmetic* (c≥4 is eager) but cannot answer their question, which is what happens
    with a capture width that actually covers 4·c. That needs a wide-capture arm — which is what
    `cgsize2` was for, and `cgsize2` was vacuous. **#19 stays owed; re-run the wide arm.**

    **One loose end, flagged and attributed to nothing:** 2 garbage streams, both in PIECEWISE c=16
    (piece1 rep1, piece3 rep0), 0 in the NONE arms. Small n (6 reps per mode), and since c=16 runs
    eager in *both* modes a cudagraph cause is implausible — so this is most likely unrelated. Do not
    quote it as a PIECEWISE defect without a reproduction.

170. **`thr` (RADIX_THRESHOLD sweep) is VACUOUS — my third self-inflicted vacuous A/B in this
    lineage (2026-09-08, raw `notes/data/thr.txt`).** Three arms, `RADIX_THRESHOLD` ∈ {16384, 20480,
    22016}, each rebuilt from source, each `FAILS: 0`, three bench starts per arm. The det column is
    **identical to the decimal across all three arms** on every cell.

    The reason is arithmetic I should have done before launching. The dispatch is
    `seq_len <= RADIX_THRESHOLD` → single-CTA. `bench_det.py`'s case list contains only
    n ∈ {1024, 4096, 8192, 16384, 32768, 65536}, and my grep kept n ∈ {16384, 32768}:
    - **16384 ≤ 16384, 20480 and 22016** → single-CTA in all three arms.
    - **32768 > 16384, 20480 and 22016** → multi-CTA in all three arms.

    No measured width changes path under any threshold I tested, so the three arms are the same
    program. **The knob could not reach the cells.**

    **The mechanism check I ran was the wrong one.** I verified the *constant* changed
    (`grep -o 'RADIX_THRESHOLD = [0-9]*'` on each arm's source, which passed). A mechanism check has
    to establish that the knob can influence *the thing being measured*, not that the knob moved.
    Same failure as `cgsize2` (no cudagraphs captured in either arm) and as the c≥4 cells of det-169
    (both arms eager). Three instances now, all mine, all in a week.

    **Standing rule, added to `notes/method.md`:** before launching an A/B, name the cell where the
    two arms must differ *and why*, in the runner's own header comment. If no measured cell can
    differ, the run is not worth the box time.

    **What the rerun needs.** Widths that straddle the thresholds, which `bench_det.py` does not
    contain and which must be added: n = 17408, 20480, 21504 flip path between these three
    thresholds; n = 16384 (always single) and 24576, 32768 (always multi) are the controls. Upper
    bound on any threshold is the caching limit — `det_select_row` caches while
    `fixed(4256) + 4n <= 101376`, i.e. **n <= 24280** — so 22016 is the largest safe value and
    24576 would silently fall to the uncached path.

    Queued as `thr2` behind `zsign`. Until it reports, **the PR body's Limitations bullet stands
    unchanged and unchallenged**: "raising it back to 32,768 is not a fix — that costs 60–100 % at
    n = 24,576–32,768 on 1–8 rows". Nothing in this run bears on it either way.

171. **BLOCKED 4-ITEM EMISSION: 25/25 cells faster, the deterministic kernel is now FASTER than
    stock on 16 of 25 (2026-09-08, `blkem`, 2 builds × 3 interleaved bench starts, raw
    `notes/data/blkem.txt`).** Branch `perf/topk-blocked-emission` in `~/git/vllm-topk-det`. The
    change three of four independent design opinions named first (det-167): give each thread four
    *consecutive* indices instead of one, so a tile is 4·N_THREADS and the emission runs one
    `cub::BlockScan` + `__syncthreads()` per 4,096 elements rather than per 1,024. Blocked (not
    striped) ownership is what keeps `pos = g + min(e, fin)` valid unchanged — it is a pure function
    of index, pivot and `fin`.

    **Correctness gate first: `FAILS: 0` on both arms.** As predicted, the selected set and its order
    are untouched; this is a pure barrier-count change.

    | | base (PR head) | blocked | worst cell |
    | --- | --- | --- | --- |
    | ratio vs stock, 25 cells | 1.01 – **1.54×** | **0.72 – 1.25×** | 1.54× → **1.25×** |
    | cells at or below 1.00× (faster than stock) | 1 | **16** | |
    | per-cell improvement | — | **9.3 – 38.3 %, all 25 same sign** | |

    Largest gains where the barrier count is highest: 1×16,384 −34 %, 16×16,384 −37 %,
    32×16,384 −38 %, 64×16,384 −37 %. The 64×32,768 cell that the PR body flags goes 75.9 → 59.6 µs.

    **Control.** `bench_det.py`'s stock column is unchanged code compiled into both builds. It
    fails to overlap in 4 of 25 cells, by **at most 4.5 %** — so the noise floor is real but the
    smallest det effect (9.3 %) is 2× the largest control drift and the bulk are 20–38 %. Arms were
    interleaved base/blk per start, 3 starts each, `.cuh` md5s recorded in the results file.

    **What this changes.** det-168 established that this kernel costs nothing end to end, so this is
    **review hygiene, not throughput** — but it is much better hygiene than expected. The PR's cost
    table stops being a liability: "1.3–4.3× on the shape grid, 1.54× worst" becomes "faster than
    stock on most shapes, 1.25× worst". Do **not** re-quote the end-to-end story on the back of
    this — it was already inside the start-to-start band, and 0.72× will not be visible either.

    Not yet measured: the full 81-shape correctness sweep on the blocked path (only `test_det.py`'s
    grid ran), and whether the `uint4` blocked load helps or the barrier reduction is the whole
    effect. Both are cheap and neither gates the change.

172. **RADIX_THRESHOLD 16384 IS TOO LOW: the single-CTA select wins the whole 16k–22k band, at every
    row count (2026-09-08, `thr2`, 3 thresholds × 3 bench starts, raw `notes/data/thr2.txt`).**
    The rerun of the vacuous `thr` (det-170), this time with widths that straddle the thresholds.
    Built on the PR head, **not** on the blocked emission — the two effects are independent and were
    measured separately. All arms `FAILS: 0`.

    det µs (min of 3 starts). S = single-CTA select, M = multi-CTA cooperative radix:

    | rows | n | thr=16384 | thr=20480 | thr=22016 | routing | multi costs |
    | --- | --- | --- | --- | --- | --- | --- |
    | 64 | 17408 | 53.3 | **34.8** | **34.8** | M/S/S | **+53 %** |
    | 64 | 20480 | 57.4 | **38.8** | **38.8** | M/S/S | **+48 %** |
    | 64 | 21504 | 59.5 | 59.5 | **39.4** | M/M/S | **+51 %** |
    | 8 | 17408 | 22.6 | **18.5** | **18.5** | M/S/S | +22 % |
    | 8 | 20480 | 24.7 | **20.6** | **20.6** | M/S/S | +20 % |
    | 1 | 17408 | 19.1 | **16.5** | **16.5** | M/S/S | +16 % |

    **Controls hold.** n=16384 (single-CTA under all three) is flat across arms; n=24576 and n=32768
    (multi under all three) are flat to the decimal. The effect appears only in the cells whose
    routing actually flips, which is what det-170's rule demands.

    **Conclusion: raise the threshold to 22016.** Every width in 16384 < n ≤ 22016 is currently
    routed to a path that costs 16–53 % more, worst at 64 rows. 22016 is the largest value that
    keeps the row cached — `fixed(4256) + 4n ≤ 101376` ⇒ n ≤ 24280 — and 24576 would silently drop
    to the uncached path, which is very likely the source of the PR body's "raising it back to
    32,768 costs 60–100 % at n = 24,576–32,768 on 1–8 rows". **That bullet stays true; it just does
    not follow that 16384 is right.** Both can hold: 32768 is too high *and* 16384 is too low.

    **This does not touch the 1.54× cell.** n=32768 is multi-CTA under every legal threshold
    (the caching limit forbids anything above 24280), so that cell is unreachable by this knob —
    exactly as det-170 predicted. What fixes it is the blocked emission (det-171: 75.9 → 59.6 µs).

    Not measured: the two changes combined, and whether a threshold between 22016 and 24280 gains
    anything more.

173. **THE TWO FIXES COMPOSE, AND det-171'S "WORST 1.25×" WAS GRID-DEPENDENT (2026-09-08, `comb`,
    4 arms × 3 interleaved starts, raw `notes/data/comb.txt`).** base (PR head) / blk (blocked
    emission, det-171) / thr (RADIX_THRESHOLD 22016, det-172) / both. **All four arms `FAILS: 0`** —
    the correctness gate ran before any benching.

    **They compose: 0 of 48 cells where `both` is worse than the better of `blk` and `thr`.** `both`
    is the best arm on every cell. So the PR can take both changes, not one.

    | arm | ratio vs stock (43 clean cells) | worst cell | at/below 1.00× |
    | --- | --- | --- | --- |
    | base | 1.01 – 2.13× | 2.13× | 0/43 |
    | blk | 0.72 – 1.79× | 1.79× | 10/43 |
    | thr | 1.01 – 2.13× | 2.13× | 0/43 |
    | **both** | **0.72 – 1.78×** | **1.78×** | **27/43** |

    `thr`'s controls are exact: n = 16,384 / 24,576 / 32,768 move **+0.0 %** at every row count. It
    changes only 16,384 < n ≤ 22,016, as designed — 64×17,408 53.3 → 34.8 µs, 64×20,480 57.5 → 38.7,
    64×21,504 59.5 → 39.0.

    **Correction to det-171.** I reported its worst cell as 1.25×. That was on `bench_det.py`'s stock
    width list (8k/16k/32k/64k). This grid adds 17,408 / 20,480 / 21,504 / 24,576, and **base's true
    worst is 2.13× at 64 × 24,576**, which `both` only brings to 1.78×. det-171's number was not
    wrong for its grid; it was quoted as if it were the worst cell of the kernel, and it is not. The
    honest headline for the PR is **"worst 1.78×, and faster than stock on 27 of 43 cells"**, not
    1.25×.

    **Why 24,576 is now the worst cell.** It is just above the 22,016 threshold, so it is multi-CTA,
    and it is above the caching bound (n ≤ 24,280) — no legal threshold can rescue it. `blk` helps
    (65.7 → 55.1 µs) but the multi-CTA path is simply expensive there. That cell is the honest
    remaining weakness of this PR.

    **Control, stated because it partly failed.** `bench_det.py`'s stock column is vLLM's installed
    `_C`, identical for all four builds. Median spread across arms 0.5 %, but **3 of 48 cells exceed
    10 %** — (1, 8192, 512) at 60.9 %, (1, 8192, 2048) at 27.2 %, (1, 17408, 512) at 15.2 % — all at
    6–14 µs where a ~4 µs timer jitter dominates. The 5 cells above 5 % are excluded from the ratio
    table above; the det-column comparison (arm vs arm) needs no stock at all and uses all 48.

174. **PREFIX CACHING WORKS UNDER MTP — the "reuse will be disabled" warning does not describe our
    path (2026-09-08, `pcx3`, 2 server arms, raw `notes/data/pcx.txt`).** All six `cgnone2` arm logs
    carry `Speculative decoding (method=mtp) is enabled but no KV cache group could be identified as
    the draft model's ... prefix-cache reuse across requests will be disabled and any external KV
    offload tier will store without ever serving a hit`. det-169 and
    `notes/data/cgnone2-contamination.md` both explain the rep=0-fast / rep=1-slow structure as
    prefix-cache warm state, which the warning appeared to contradict. Measured, three identical
    sequential 6,946-token requests per arm, `vllm:prefix_cache_hits_total` deltas
    (`usage.cached_tokens` is inert here and indeed reports `absent`):

    | arm | warning | rep 0 | rep 1 | rep 2 |
    | --- | --- | --- | --- | --- |
    | `mtp3` (FN_MTP=3) | **fires ×2** | 3.52 s, 0 hits, 0.0 % | 0.71 s, 6400 hits, **92.1 %** | 0.71 s, 6400 hits, 92.1 % |
    | `nospec` (FN_MTP=0) | absent | 4.29 s, 0 hits, 0.0 % | 0.94 s, 6272 hits, **90.3 %** | 0.86 s, 6272 hits, 90.3 % |

    **Both arms hit, and MTP hits slightly MORE, not less.** The mechanism is exact: the hit count is
    4 full blocks in both arms — 4 × 1600 = 6400 with MTP's attention block, 4 × 1568 = 6272 without
    — with the 546/674-token partial tail uncached, as prefix caching must behave. The only
    difference between the arms is the block size, not the spec config.

    **Consequences.**
    - **det-169 needs no correction**, and the retraction in `cgnone2-contamination.md` is now
      *measured* rather than argued from the shape of the numbers: rep 0 takes 0 hits at 3.5–4.3 s,
      reps 1–2 take 6400/6272 hits at 0.7–0.9 s. That 5× TTFT drop is exactly the rep structure.
    - Memory `prefix-cache-works-agent-loop` is **confirmed**, not contradicted.
    - The warning's operative clause is about an **external KV offload tier** serving hits. We run
      none, so that half is moot for us, and the prefix-cache half of the sentence overstates what
      happens on this path. Do not read that log line as "our cache is off" again.
    - **This is the PRE-#52771 state.** Our venv is `8340fe1bb` (2026-09-04); vllm#52771
      (`4a806d08e`, merged 09-07 12:16) fixes the all-groups drafter fallback that emits this
      warning, and we run `FN_SPEC_NODROP=1`, its precondition. So the fix is still one we are
      missing — but the degradation I feared it was causing in our benchmarks **did not happen**.

    **Method note, because it cost two runs.** The first attempt sent a 22.5k-token prompt against
    `FN_MAXLEN=16384` (six 400s); the second crashed in post-processing on
    `usage.prompt_tokens_details` being `null` *after* the requests had already succeeded — a
    cosmetic field discarded six good measurements. The probe now prints timing and counters before
    touching any optional field, and the parse was validated offline against all four response
    shapes. Two 25-minute runs were spent on my own parse bugs, not on the box.

175. **vllm#55872's opt-in FlashInfer TopK backend DOES NOT RUN on sm_121 (2026-09-08, `pr872`, raw
    `notes/data/pr872.txt`).** LopezCastroRoberto reviewed our PR #55122 on 08 Sep, argued against
    changing the default TopK, opened #55872 (`--dsa-topk-backend native|flashinfer` plus a tie-break
    policy) and asked us directly to test it — we are the only GB10 in that thread.

    Their patch applies **cleanly** to dev524: 7 files, 0 failed hunks, including
    `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`, which our det overlay also patches.
    `sparse_attn_topk` imports, and the API it calls (`top_k_ragged_transform`, `TopKTieBreak`) is
    present in **flashinfer 0.6.17** — so their backend needs no 0.6.18 bump.

    **But the engine never starts with the backend enabled.** `--dsa-topk-backend flashinfer
    --dsa-topk-tie-break small`, cold load, dies at 720 s during engine init:

        RuntimeError: Check failed: (status == cudaSuccess) is false:
          TopKRaggedTransform failed with error code operation not supported

    `operation not supported` from a CUDA call at init is the signature of a kernel with no sm_121
    implementation. The `native` arm on the same patched build started fine, so this is the backend
    itself, not the patch or our overlay.

    **Why this matters to the #55122 review, stated carefully.** It does not make their design wrong
    — an opt-in backend is a reasonable shape, and this may well be a fixable FlashInfer gap. What it
    does mean is that *today*, on sm_121, their backend cannot be the deterministic option, so it
    does not yet substitute for a deterministic native kernel on this hardware. That is a fact about
    availability, not about which approach is better, and it should be reported to them that way.

    **The determinism comparison in this run is VOID.** Arms `native` and `ours` both returned
    **0/8 completions — empty `content` AND empty `reasoning_content`** — so there are no hashes to
    compare and nothing can be concluded about either. `fn3smoke` hit the same thing on fnmain3
    (`request: None`). Two candidate causes, not yet separated:
    (a) my probe reads the wrong response field for this build, or
    (b) **fnmain3 itself emits no output** — i.e. my hand-port of the PLE hunks is broken in a way
        that loads, serves 200s, and produces nothing.
    (b) would be a serious defect in today's venv bump and must be excluded before fnmain3 is used
    for anything. The discriminator is one identical request against fnmain2 (known good) and
    fnmain3, dumping the raw JSON rather than a parsed field. Queued as `emptydiag`.

    Three probe iterations have now been lost to guessing at this response shape. Dump the raw
    response before touching the parser again.

176. **The PLE hand-port on fnmain3 VERIFIES against the known-good venv — so the empty completions
    are not (yet) explained by the bump (2026-09-08).** det-175 left two candidates for the 0/8 empty
    completions: (a) my probe reads the wrong response field, or (b) fnmain3's hand-ported PLE hunks
    are broken. Checked (b) directly instead of inferring it from an end-to-end run, using fnmain2
    (dev401, same overlay, known good) as the reference and comparing normalised ASTs so comments and
    formatting cannot mask a difference:

    | site | fnmain2 vs fnmain3 |
    | --- | --- |
    | classes / methods on the whole module | none dropped, none added |
    | `load_weights` (the offload branch) | **identical** |
    | `get_offload_output_dim`, `get_offload_output_dtype`, `initialize_dummy_offload_metadata` | **identical** |
    | `Qwen4ExpPLELayer.__init__` construction site | **identical** — `torch.device(PleOffloadLayer.get_target_device())`, `_offload_quant_method`, and the `VLLM_PLE_CPU_OFFLOAD and not is_offload_process()` guard |
    | `forward_impl` | one intended difference only (below) |

    The `forward_impl` difference is the rebase and nothing else: fnmain2's non-offload branch builds
    `ngram_ids` with `input_ids.new_empty(...)` and the `qwen4_exp_compute_ple_ngram_ids` custom op;
    fnmain3 calls `self.compute_ngram_ids(...)` hoisted above the branch. vllm#55272 deleted that
    custom op, and upstream's own dev524 `forward` is exactly `compute_ngram_ids` then embed — so the
    port matches upstream semantics rather than inventing a substitute.

    **Conclusion: hypothesis (b) is not supported.** The port is sound at the level a static check can
    reach, which shifts suspicion to the probe or the prompt. `emptydiag` still runs, but its job is
    now to characterise the RESPONSE SHAPE (message keys, `finish_reason`, `usage`) on both builds,
    with fnmain2 as the control — if fnmain2 also returns empty, the bump is exonerated outright.

    **Method note:** the user asked "shouldn't you verify PLE first?" — correct, and cheaper. Two cold
    server loads (~30 min) would have told me *whether* fnmain3 was broken; a five-minute AST diff
    against the known-good venv told me *what changed*, which is the question that actually mattered.
    Verify the thing you changed against a reference before inferring it from end-to-end behaviour.

177. **fnmain3 (dev524 + the hand-ported overlay) IS PROVED WORKING (2026-09-08, `emptydiag`, raw
    `notes/data/emptydiag.txt`).** Same three request shapes against fnmain2 (dev401, known good) as
    control and fnmain3, dumping raw JSON rather than a parsed field. All five criteria:

    1. **Starts, and the log names the right venv** — `/opt/llm/runtime/vllm-venv-fnmain3`, the
       `venv-copy-shebang-trap` proof (fn3smoke).
    2. **Static** — 0 `.rej`, 17 overlay files compile, 5 markers, registry True, PLE port
       AST-identical to fnmain2 bar the intended rebase (det-176).
    3. **Real content**, all three shapes, `finish_reason: stop`.
    4. **Equivalence with the control:**

       | shape | fnmain2 | fnmain3 |
       | --- | --- | --- |
       | short | `\n\nHello!` | **byte-identical** |
       | effort | `\n\nHello! 👋 How are you doing today? …` | **byte-identical** |
       | long, 3,018 tok | coherent summary | coherent summary, different wording |

       `system_fingerprint` confirms `dev401+g8340fe1bb` vs `dev524+g5db652225`, so these really are
       different builds. The long-shape wording difference is **not** evidence of a defect: the two
       builds differ by 123 dev revisions including the removal of torch.compile for this model, and
       memory `temp0-not-reproducible-under-load` records that Flash-Next diverges at temperature 0
       from ~30 tokens even on one build. Two byte-identical shapes is the strong result here.
    5. **PLE offload exercised, not merely present** — `PleOffload: spawning worker (rank=0 …
       ipc://…)` and a separate `PleOffloadWorker` process, 238 log lines, no fallback. The only
       `ERROR` lines are 8 copies of a `Qwen3VLVideoProcessorInitKwargs` docstring complaint from
       transformers, present on both builds and unrelated.

    **And the empty completions that started this are fully explained — no defect anywhere.**
    Two independent mistakes of mine stacked:
    - **Wrong field name.** The reasoning text is in `message.reasoning`, **not**
      `reasoning_content`. My fallback could never match, so an empty `content` printed as "EMPTY
      content AND reasoning_content" and looked like a dead server.
    - **Reasoning ate the token budget.** `completion_tokens_details.reasoning_tokens` is 26 for even
      "Say hello." `fn3smoke` asked for `max_tokens=24` — the budget was spent thinking before any
      content existed. `pr872` asked for 160 on a harder summarise task whose reasoning ran 108–123
      here, leaving little or nothing.

    **Consequence for det-175:** its FlashInfer-backend result stands untouched (the engine never
    started, so no probe was involved), but its two "candidate causes" for the empty arms are now
    closed — it was the probe, and fnmain3 was healthy the whole time. The `pr872` determinism
    comparison still needs re-running with a sane token budget and the right field.

    **Method:** the user's "shouldn't you verify PLE first?" was the cheaper and better order, and
    the raw-JSON dump is what actually settled it. Three probe iterations were spent guessing at a
    response shape that one dump revealed.

178. **ZC502's CLIENT collector WORKS on sm_121 — but our test case did not engage the defect, so the
    determinism comparison in it is VOID (2026-09-08, `vpp4b`, raw `notes/data/vpp4.txt`, reports in
    `notes/data/vpp4-report4/`).** On #54521 ZC502 replied to our blocker (det-155: the offline
    collector constructs `LLM()` in-process and cannot load Flash-Next on GB10) by shipping
    `collect_client.py`, a stdlib-only client that POSTs to a running server, and asked for a first
    live validation.

    **Primary answer: it works.** 12 of 12 collector runs `exit=0` on GB10 / sm_121 — two arms
    (`FN_DET_TOPK` 0 and 1) × three prompt lengths (1,460 / 1,999 / 5,960 tokens) × sequential and
    concurrent — producing 7.5–31 MB canonical JSON each, and `analyze.py` consumes them and emits
    reports. The blocker is genuinely gone: no launch flags to reproduce, no in-process model load,
    no PLE memory problem.

    **But the determinism result is void, and this is the falsification condition stated before the
    run:**

    | case | det0 = stock, disagreeing positions | det1 = #55122 | cross-arm mismatch |
    | --- | --- | --- | --- |
    | 1,460 / 1,999 / 5,960 **sequential** | **0 / 0 / 0** | 0 / 0 / 0 | 0 / 0 / 0 |
    | 1,460 / 1,999 / 5,960 **concurrent** | 619 / 635 / 3,055 | **identical: 619 / 635 / 3,055** | 0 / 0 / 65 |

    **Stock is self-consistent on all three sequential cases**, so it never exhibited the bug and
    det1's cleanliness demonstrates nothing about the kernel. The concurrent columns are *identical
    between arms* — same disagreeing-position counts, same max forced-logprob spread (1.35 / 0.987 /
    1.16), same first position — which is the GDN batch-invariance effect (finding 76), not the
    top-k, and must not be read as a #55122 failure.

    **Why the case missed.** The prompts are `random.choice` over a 10-word vocabulary, inherited from
    the earlier offline harness. That is a poor generator for this defect: the bug needs ties at the
    top-k boundary, and #54521's original reproduction used real text near `indexer_budget`. Length
    alone was not the discriminator — 5,960 tokens is well above the 2,048 budget and still clean
    sequentially.

    **Next:** re-run with prompts that actually tie — real prose near the budget, plus a deliberately
    tie-heavy case — before offering ZC502 any det0/det1 numbers. The collector validation stands on
    its own and can be reported now.

    **One usability note for them:** `analyze.py` takes `reference [candidate]` as JSON *files* with
    `--out` a directory; passing the output directory positionally gives
    `IsADirectoryError: Is a directory`. Cost me two attempts. Worth a line in the README.

179. **A fifth void run — and the pattern across it and det-178 is itself the finding: we cannot
    reproduce the top-k defect end-to-end in THIS server configuration (2026-09-08, `vpp5`, raw
    `notes/data/vpp5.txt`).** det-178 blamed random-word prompts for not tying at the top-k boundary,
    so this run held length fixed and varied tie *shape*: one sentence repeated (maximal identical
    keys), natural prose, and a cycled phrase set. Sequential only, 8 repeats, both arms.

    | tie shape | det0 = stock | det1 = #55122 | cross-arm |
    | --- | --- | --- | --- |
    | `repeat` | **0** disagreeing positions | 0 | 0 |
    | `prose` | **0** | 0 | 0 |
    | `clustered` | **0** | 0 | 0 |

    The two arms' collector files are byte-identical per shape (10,396,079 / 10,615,797 / 10,510,501),
    so stock and our kernel produced the same output, position for position.

    **My own error, stated first: the prompts were not above the budget.** I sized them by
    `chars/4`; English prose here tokenizes at **5.52 chars/token**, so 10,439 characters gave
    1,874–1,896 tokens — *below* the 2,048 `indexer_budget`, not the ~2,600 the runner's header
    claims. So this run did not test what it said it tested, and its header is wrong.

    **But that does not rescue it**, because det-178 already covered above-budget: 5,960 tokens,
    sequential, also 0. Taken together that is **six sequential cases from 1,460 to 5,960 tokens,
    two prompt generators, zero divergence in stock.**

    **The reconciliation, and the next test.** The kernel *is* non-deterministic — det-151 measures
    0/81 shapes self-consistent at the kernel level, and finding 76 records k3dani seeing 0/4 prompts
    reproducible end to end. The difference is configuration: **k3dani ran prefix caching ON, chunked
    prefill, MTP=2, PIECEWISE graphs**, and our own finding 82 saw stock take a different trajectory
    under MTP n=5. Every `vpp` run has prefix caching **off** and **no speculation** — chosen to make
    the case clean, which appears to have removed the very conditions that expose it.

    So the hypothesis to test next is not another prompt: it is **prefix caching ON + MTP**, the
    configuration in which divergence has actually been observed here and by a third party. If stock
    diverges there and not with the cache off, that is a sharper statement about the defect than any
    prompt shape — and it is what we should give ZC502.

    **Method note.** Five void runs today (cgsize2, cgnone2 c≥4, thr, vpp4, vpp5). The rule in
    `method.md` says name the differing cell; this run adds a second clause worth writing down:
    **verify the control arm actually misbehaves before trusting the treated arm's cleanliness** —
    and check the units you claim (tokens, not characters) with the tokenizer, not an estimate.

180. **The "stock" arm in every `vpp` run was not stock — it carried three of the four determinism
    fixes. That makes det-178/179's nulls a REAL result, differently labelled (2026-09-08, `vpp6`,
    raw `notes/data/vpp6.txt`).**

    `vpp6` changed the one variable det-179 pointed at — **prefix caching ON, MTP=3**, prompts
    genuinely above the budget this time (2,447 / 2,496 tokens; the mechanism line confirms
    `enable_prefix_caching=True`, `spec=method='mtp'`). Result: both arms **0 disagreeing positions,
    max spread 0.0**, collector files byte-identical per shape. Eight sequential cases now, across
    two configurations, with no divergence.

    **Then I checked what `FN_DET_TOPK=0` actually disables.** `prod_det_overlays.sh` installs four
    fixes, and only two are env-gated:

    | fix | gate | state in our "det0" arm |
    | --- | --- | --- |
    | `qsadet` — deterministic `persistent_topk` | `VLLM_QSA_DET_TOPK` | **off** (what we intended) |
    | `detfin` — bit-stable MoE finalize | `VLLM_MOE_DET_FINALIZE`, defaults **on** | **on** |
    | `moe_cachekey` — FlashInfer autotune cache key | **not env-gated** | **on** |
    | `plefix` — PLE offload semaphore reset | **not env-gated** | **on** |

    So `det0` was stock-top-k **plus** the other three, including the PLE semaphore fix that
    finding 138 called "the residual noise of four days". I labelled the arm "stock" in three runners
    and in det-178/179 without checking, which is the same class of error as calling a run's knob
    verified because the constant changed.

    **What the eight cases therefore actually establish** — and it is worth more than what I set out
    to measure: **with the MoE finalize, autotune cache-key and PLE semaphore fixes in place, the
    top-k fix alone is not required for sequential position-level reproducibility on real prompts**,
    at 1,460–5,960 tokens, cache on or off, spec on or off, across three tie shapes. That is a
    sharper statement than "we could not reproduce it", and it bears directly on
    LopezCastroRoberto's argument on #55122 that the default should not change without evidence of
    user-visible harm — it is *evidence on their side*, from us, and we should say so.

    It does **not** contradict det-151 (the kernel is 0/81 self-consistent on synthetic tie-heavy
    inputs) or finding 76 (k3dani saw 0/4 prompts reproducible on the preview image, before any of
    these fixes existed). Both remain true. The reconciliation is that the top-k defect needs ties at
    the selection boundary, and **real prompt score distributions may simply not produce them** —
    which is what #53287 concluded independently.

    **The next test is now well-posed:** a *true* stock arm — `FN_DET_TOPK=0`, `FN_DET_FINALIZE=0`,
    **and** `prod_det_overlays.sh off` to remove the two ungated patches — against the full-fix arm.
    That isolates whether any of the four is load-bearing here, and it is the run that should have
    been done first.

181. **TRUE STOCK IS BADLY NON-REPRODUCIBLE END TO END; ALL FOUR FIXES TOGETHER MAKE IT EXACT
    (2026-09-08, `vpp7`, raw `notes/data/vpp7.txt`).** The comparison det-180 said should have come
    first: `truestock` = `prod_det_overlays.sh off` **and** `FN_DET_TOPK=0` **and**
    `FN_DET_FINALIZE=0` (none of the four fixes) against `allfixes` (all four). Prefix caching ON,
    MTP=3, prompts 2,447 / 2,504 tokens, sequential, 8 repeats. Mechanism check on the stock arm:
    `QSADET=0`, `enable_prefix_caching=True`, `spec=method='mtp'`.

    | prompt | truestock | allfixes | cross-arm modal-top1 mismatches |
    | --- | --- | --- | --- |
    | `repeat` | **122** disagreeing positions, max forced-logprob spread **4.51**, first at position 3 | **0**, spread 0.0 | 17 |
    | `prose` | **335** disagreeing positions, max spread **10.63**, first at position 2 | **0**, spread 0.0 | 104 |

    Divergence starts at position 2–3 — effectively immediately — and the modal top-1 token differs
    at 17 and 104 positions respectively, so this is not sub-threshold logprob noise: the model emits
    different tokens run to run.

    **This settles the eight nulls of det-178/179/180.** They were not evidence that the defect is
    unreachable; they were measured with three of the four fixes silently active. Remove all four and
    the same prompts, the same config, the same harness produce immediate, large divergence.

    **It also means the concession drafted from det-180 was wrong in its framing and must not be
    posted as written.** "The top-k fix alone was not required" remains literally true — this run
    does not isolate which fix carries the weight — but the impression it gives, that end-to-end
    reproducibility is not a real problem here, is refuted by this run. Draft revised.

    **The question this opens, and it decides how #55122 should be argued:** *which* of the four is
    load-bearing? Four one-at-a-time arms against true stock answer it:
    `qsadet` (our PR), `detfin`, `cachekey`, `plefix`. If `qsadet` alone closes it, #55122 is the
    fix and the reproducibility argument stands as originally made. If `plefix` alone closes it — the
    PLE offload semaphore, finding 138's "residual noise of four days" — then our own PR is *not* the
    load-bearing one for end-to-end reproducibility, and we should say so on the thread before anyone
    merges anything on our account. Queued as `isolate4`.

    **Venv hygiene held.** The EXIT trap restored the overlays: `state: qsadet=1 detfin=1 cachekey=1
    plefix=1` in the results file. Worth noting because a killed run here would have left fnmain2
    stripped and silently poisoned every later measurement.

182. **CORRECTION to det-180: upstream does NOT ship `vllm/v1/ple_offload/`, and the committed
    dev524 overlay was incomplete (2026-09-08).** Asked whether the PLE offload semaphore fix is
    upstream. It is not, and checking exposed an error in my own artefact.

    - **The wheel is the evidence.** `vllm-0.28.1rc1.dev524+g5db652225…whl` contains **zero** files
      under `vllm/v1/ple_offload/`, and no `ple_offload_layer.py`. Both the subsystem and our
      semaphore fix exist only here and on peakcrosser7's `release/qwen38next_offload` branch.
      **vllm#53899 is open and unmerged; our PR #13 on that branch is open and unmerged.**
    - **Why I got it wrong.** I regenerated the overlay by diffing fnmain3 against
      `…-pristine-dev524.tgz`, taken right after `pip install --no-deps`. But fnmain3 is a *clone of
      fnmain2*, and `pip install` only overwrites files the wheel contains. Our five added files are
      not in the wheel, so they survived into the "pristine" snapshot, sat identically on both sides
      of the diff, and disappeared from the overlay. I then wrote the disappearance up as "upstream
      now ships them" — an inference from an artefact I had built wrong, exactly the shape of error
      det-176 was supposed to have taught me to check.
    - **Consequence:** `tools/main/fnmain-overlay-dev524.diff` covered 12 files and would have
      produced a venv with **no PLE offload subsystem at all**. Now 17 files, with the five added
      ones appended as new-file hunks. A full regeneration against the extracted *wheel* is queued;
      the rule is in `BUILD-RECIPE.md`: **a `--no-deps` install is not a reset, it is an overwrite of
      the intersection — diff against the wheel, never against a tarball of a populated venv.**

    **Why this matters beyond hygiene.** If `isolate4` shows `plefix` is the load-bearing fix for
    end-to-end reproducibility, then the thing that makes greedy decoding reproducible on this stack
    is a patch that exists in **no released vLLM and no merged PR** — sitting on a fork branch behind
    an unmerged feature PR. That would be the single most useful thing we could tell #54521 and
    #55122, and it would be more important than our own kernel PR.

183. **NO SINGLE FIX CLOSES IT — the four are jointly necessary, not individually sufficient
    (2026-09-08, `isolate4`, raw `notes/data/isolate4.txt`). Read with the two defects below.**
    One fix active at a time against a no-fix control, prefix caching ON, MTP=3, 2,504-token prose
    prompt, 8 sequential repeats:

    | arm | disagreeing positions | max forced-logprob spread | first divergence |
    | --- | --- | --- | --- |
    | `none` (control) | 325 | 7.88 | position 1 |
    | `qsadet` — **our PR #55122** | 348 | 7.25 | position 1 |
    | `cachekey` | 311 | 9.81 | position 4 |
    | `plefix` | 282 | 10.83 | position 2 |
    | `detfin` | **arm died** | — | — |

    The control misbehaved, so the run is valid. **No arm comes near zero**, while all four together
    give exactly 0 (det-181). The spread between arms (282–348) is one run each and should not be
    read as a ranking — MTP trajectories differ by more than that between restarts (memory
    `mtp-restart-instability`, up to 1.83×).

    **So the honest statement for #55122 is: our kernel is one necessary component of a set, not the
    fix.** The body's "Fixes #54521" and "one of three independent defects that together make
    Qwen3.8-Flash-Next reproducible" are both wrong in the same direction — the second because the
    set is four and includes one that is not upstream at all (det-182).

    **Two defects in this run, both predicted by our own notes:**
    1. **`detfin` alone is not a runnable configuration.** It died at engine init with
       `Invalid gemm2 profile id: 59` — verbatim the failure in memory
       `spec-compile-cache-key-omits-nspec`: flipping `use_fused_finalize` invalidates the FlashInfer
       autotune cache while the cache key ignores that flag. **That is precisely what `cachekey`
       fixes**, so `detfin` can only be tested paired with it. The four fixes are not independent.
    2. **All five arms shared one `FN_CACHE_ROOT`.** The same note says per-arm roots plus a purge.
       `none` and `qsadet` both ran `finalize=0` so their cache state was mutually valid, and the
       top-k change is a `.so` swap rather than a compiled artefact — but "probably unaffected" is
       not the standard for a number that would go upstream.

    **Therefore the headline of this finding is provisional.** It needs `isolate5`: per-arm
    `FN_CACHE_ROOT` with a purge between arms, `detfin` tested as `detfin+cachekey`, and `none` and
    `qsadet` redone on that footing. Until then, quote det-181 (all four vs none, which was a clean
    two-arm comparison) rather than these per-arm numbers.


184. **isolate5 confirms det-183 on a valid footing: no single fix removes even a fifth of the divergence, and all
    four together remove all of it (2026-09-08, `isolate5`, raw `notes/data/isolate5.txt`).** The three defects det-183
    listed are all fixed here: per-arm `FN_CACHE_ROOT` purged between arms, `detfin` tested as `detfin+cachekey` (it
    cannot run alone), and every arm re-measured on that footing. Same cell as before — prefix caching ON, MTP=3,
    2,504-token prose prompt, 8 sequential repeats, disagreeing positions against the arm's own repeats:

    | arm | disagreeing positions | max forced-logprob spread | first divergence |
    | --- | --- | --- | --- |
    | `none` (control) | **333** | 8.79 | position 2 |
    | `qsadet` — our PR #55122 | 330 | 8.54 | position 4 |
    | `cachekey` | 334 | 9.01 | position 1 |
    | `plefix` | 285 | 9.05 | position 6 |
    | `detfin`+`cachekey` | 280 | 8.12 | position 1 |
    | **all four** | **0** | 0.0 | — |

    Validity gate, pre-committed before the run and passed: the control must misbehave (333 ≠ 0) and the positive
    control must not (0). Stock is now measured fix-free three times — 325 (isolate4), 333 (isolate5), 335 (det-181) —
    which is the tightest thing in this whole investigation and says the cell itself is stable.

    **Reading.** Every single-fix arm sits between 280 and 334 against a control of 333: the best of them removes 16 %
    of the disagreeing positions, our own kernel removes 0.9 %, and one arm is *above* the control. One run each, so
    280 vs 334 is not a ranking (MTP restart spread reaches 1.83×, memory `mtp-restart-instability`) — but the
    qualitative statement does not depend on the ranking, because the gap between "any one fix" and "all four" is not
    a matter of degree: 280…334 versus 0. **The four defects are jointly necessary and individually almost worthless.**
    That is an unusual shape and it is worth saying plainly: each defect alone is enough to destroy reproducibility, so
    removing three of four buys nothing a user can observe.

    **How far apart the two builds actually are, not just how unstable one of them is.** Running ZC502's `analyze.py`
    in its two-file form over the same traces (`none` as reference, `all4` as candidate) gives a cross-arm block the
    per-arm numbers do not: **110 of 2,504 positions (4.4 %) differ in their MODAL top-1 token**, first at position 2,
    with a maximum absolute difference in mean forced logprob of 5.89. So the defects do not merely make stock jitter
    between runs — they move the answer stock converges on, at one position in twenty-three. That is the number to
    quote when someone asks whether determinism work changes output quality or only reproducibility.

    This is the evidence behind the correction already posted to #55122 (`issuecomment-5590003708`) — that our kernel
    is one necessary component of a set rather than the fix, and that `Fixes #54521` had to go. det-183's headline
    stands; only its per-arm numbers are superseded by the table above. **The determinism divert ends here**; the goal
    is agent turn time again.


185. **mmastrac's tool-call-corruption repro fires far harder here than on the machine it was written for, and our four
    fixes close it completely: stock gives 40 distinct completions from 40 identical greedy requests, fixed gives 1
    (`tcorrupt`, `notes/data/tcorrupt.txt`, 2026-09-09).** They posted the repro on vllm#54521 (gist
    `ff0d09589d5480f8614413daf6b1552b`, 2026-09-08 19:56 UTC) as an explicit "this might be useful to see if it
    reproduces in other models and configurations". Script AST-reviewed before running: stdlib only, one network call
    to `--url`, no file writes, no eval/exec/pickle — the `subprocess.run` text inside it is a string constant, the fake
    repository file the synthetic transcript pretends the agent read.

    Cell: Qwen3.8-Flash-Next NVFP4, GB10 sm_121, **TP=1**, MTP-3, `FN_MAXLEN=65536`, the synthetic 47-tool coding-agent
    transcript at **49,902 prompt tokens**, 40 runs at temperature 0 with a fixed seed, prefix caching on as in their
    setup.

    | arm | distinct completions / 40 | runs diverging from the majority |
    | --- | --- | --- |
    | **stock** (all four fixes off) | **40** | 39 |
    | **fixed** (all four on) | **1** | 0 |

    **Their result on 4× GB10 TP=4 with GLM-5.3-Flash-NVFP4 was 5 distinct in 40** (30 correct, then 4/3/2/1), with the
    corruption appearing at token positions 23–31. Ours is categorically worse: **every run differs, and the divergence
    starts at token 0 or token 1** — "I've" vs "I keep" vs "The" vs "Let" as the opening token. A second model, a
    different tensor-parallel degree, a different parser, the same hardware family, and the failure is not a rare
    corrupted argument but total non-reproducibility from the first token.

    Why this matters more than the throughput work it interrupted: their filing notes that a corrupted tool *name*
    makes the parser emit zero deltas, so the request finishes `stop` with no content and no tool calls and the client
    reports "completed response with no content". **That is a whole agent turn lost**, which costs more than any of
    tonight's speed levers buys — the best of them (finding 153) is −9.6 % on turn time. MiaAI-Lab report the same class
    on this checkpoint in their dual-Spark #42 ("corrupted tool-call names in long agent sessions").

    This is also the cleanest statement yet of what the four fixes are for. det-184 measured them on a 2,504-token prose
    prompt and got 333 disagreeing positions → 0. Here, at 50k tokens of realistic agent context, the same set takes
    **40 distinct completions → 1**. The fixes are not a reproducibility nicety for benchmark hygiene; on long agent
    traffic they are the difference between a deterministic server and one that answers differently every time.

    **Draft for #54521 written, NOT posted** (`notes/upstream/comment-54521-tcorrupt.md`) — needs the user's go.


186. **The concurrent nondeterminism is NOT an MTP defect: it is generic batch-shape dependence in prefill, present
    with speculation entirely off. MTP roughly doubles it but does not cause it — and the whole pattern reproduces
    exactly across server restarts (`conc1`, mtp3/mtp0 × 2 starts, each collecting sequentially AND concurrently,
    `notes/data/conc1.txt`).** Asked because both arms of finding 158 ran MTP-3, so MTP had never been excluded.

    | arm | nonzero spread | spread > 0.1 | max | top-1 flips | first spread @ |
    | --- | --- | --- | --- | --- | --- |
    | mtp3 sequential (×2) | **0 / 2,504** | 0 | 0.000000 | 0 | — |
    | mtp0 sequential (×2) | **0 / 2,504** | 0 | 0.000000 | 0 | — |
    | mtp3 concurrent (×2) | 2,503 / 2,504 | 565 (22.6 %) | 8.106 | 234 | **1** |
    | mtp0 concurrent (×2) | 2,503 / 2,504 | 455 (18.2 %) | 6.686 | 179 | **1** |

    **Answer: generic batching.** With speculation off entirely, 8 concurrent greedy repeats of the same prompt still
    perturb **every scored position**, starting at position 1. The within-arm sequential control is exactly 0 in all
    four arms, so the concurrent figure is attributable to concurrency and nothing else — not the build, the prompt or
    the server.

    **MTP amplifies by about 2×, it does not cause.** Per-position spread ratio mtp3/mtp0 has **median 2.25** over 2,502
    common positions; top-1 flips go 179 → 234; and **88.8 % of the MTP-off flipped positions are also flipped with
    MTP on** — the same underlying instability, pushed harder. That is what one expects from a knob that adds 1+k
    tokens per sequence per step: it changes batch shapes, it does not introduce a new source of variation.

    **The most useful detail is that this "nondeterminism" is deterministic.** The exact set of 234 (and 179) flipped
    positions is **identical across two independent server starts** — same cold load, same compile cache rebuild, same
    everything-from-scratch. So this is not entropy; it is a reproducible function of batch composition, and our harness
    submits the same 8 requests the same way each time. That matters twice over: it makes the effect bisectable rather
    than statistical, and it means a fix is a matter of making a reduction order batch-independent rather than of
    chasing a race.

    **Why nobody has fixed it here:** `VLLM_BATCH_INVARIANT=1` refuses to start on this model
    (`batch-invariance-unavailable.md`) — `supports_batch_invariance()` is implemented by five full-attention backends
    and no mamba/linear-attention backend, while 36 of our 48 layers are linear attention. There is no batch-invariant
    control arm to compare against, which is also why this is unmeasured elsewhere.

    **Next rung, NOT started.** `layerhash_patch.py` is *not* ready for this: it targets the old
    `vllm-venv-fnext`/`qwen3_8_flash_next` paths, it needs `--enforce-eager` (which changes the execution path we are
    trying to measure), and it hashes each layer's whole output tensor — under concurrency that tensor holds several
    requests' tokens, so a mismatch does not isolate the diverging request. It needs per-request slicing first. A
    cheaper black-box rung exists and should come before instrumentation: **vary `--max-num-batched-tokens`**. Our
    2,504-token prompt prefills in one chunk alone but eight of them get packed into 4,096-token batches, so each
    request's prompt is split differently depending on scheduling. If the perturbation tracks the chunk budget, the
    culprit is chunk packing rather than any single kernel, and that is one A/B rather than a multi-hour trace.


187. **Provenance correction to det-184's fix list and to the README (2026-09-09).** Two claims made this morning were
    wrong and are corrected here because they concern what a reader would rely on.

    **(a) #55375 is not ours.** It is peakcrosser7's, merged 2026-09-05 14:02 UTC as `28e605fb33`. We found the same
    defect independently and opened **#55467**, which I closed as a duplicate at 13:35 the same day — 27 minutes before
    theirs merged — and moved the GB10 evidence onto #55375. What is ours on that bug is the independent discovery, the
    multi-prefill reproducer and the evidence, not the merged patch. (Third duplicate of this kind; see the memory
    `search-open-prs-before-fixing`.)

    **(b) The fix is in the serving venv as our overlay, not from the wheel.** `vllm-venv-fnmain2` is built from nightly
    `0.28.1rc1.dev401+g8340fe1bb`, and `8340fe1bb9` is dated **2026-09-04 20:27** — eighteen hours *before* the merge.
    `git merge-base --is-ancestor 28e605fb33 8340fe1bb9` is false. The byte-identity check I ran this morning compared
    the venv against **our** commit `789f55ae5b`, not against upstream's merged file, and reported "fix present" — true,
    but it does not support the sentence it was used for ("merged upstream and already in the build").

    **What survives the correction:** the two implementations are semantically identical. The entire diff is the argument
    *position* of `state_idx_stride` in the two kernel signatures plus a hoisted local with a comment on our side against
    an inline `state_indices.stride(0)` on theirs. So every measurement taken on this venv stands, and a bump to any
    nightly from 2026-09-05 14:02 onward replaces our overlay with upstream's equivalent rather than conflicting with it
    — which makes the fnmain3 cutover and any future venv bump simpler, not riskier.

    **Method note.** A byte-identity check answers "is this file the version I have in hand", not "did this ship". For a
    provenance claim the test is ancestry — `git merge-base --is-ancestor <merge-commit> <build-commit>` — and it costs
    one command. Use it before writing "already in the build" anywhere a reader might act on it.


188. **The chunk budget modulates the concurrent perturbation by 2×, the curve is PEAKED not monotone, and the affected
    positions are NESTED — one mechanism scaled, not different mechanisms at different shapes (`conc2`, budgets
    2048/4096/16384 × 2 starts, each collecting sequentially and concurrently, `notes/data/conc2.txt`).**

    | `--max-num-batched-tokens` | non-zero spread | spread > 0.1 | max | top-1 flips | first spread @ |
    | --- | --- | --- | --- | --- | --- |
    | 2,048 (×2) | 2,499 / 2,504 | 290 (11.6 %) | 3.683 | 119 | 5 |
    | **4,096 (×2, control)** | 2,503 / 2,504 | **565 (22.6 %)** | 8.106 | **234** | **1** |
    | 16,384 (×2) | 2,499 / 2,504 | 330 (13.2 %) | 4.719 | 132 | 5 |

    Both controls hold: every sequential arm is exactly 0/2,504, and the 4,096 arms reproduce det-186 to the digit on a
    different day with a fresh server and rebuilt cache.

    **My pre-registered gate was wrong, and I am recording that rather than reinterpreting it.** I offered two outcomes,
    "monotone in the budget" (packing drives it) and "flat" (packing is irrelevant). The truth is a third: **peaked at
    4,096**, with both neighbours roughly half as severe. The prediction was badly posed; the run still answered.

    **Why a peak is the chunk-packing signature.** Eight requests × 2,504 tokens = 20,032. At 2,048 a forward holds less
    than one prompt, so batches are close to one request's chunk at a time. At 16,384 it holds six whole prompts, so
    prompts mostly go in intact and split points are rare. At 4,096 it holds one whole prompt **plus 1,592 tokens of the
    next** — maximally incommensurate with the prompt length, so boundaries land inside prompts in the most variable
    way. `first_spread@` agrees: divergence reaches position **1** only at 4,096. A reduction that depended merely on
    the batch's *total token count* could not produce a peak.

    **The sharper test, and it is the more informative one** (framing owed to a third-party reading of these notes,
    2026-09-09: ask whether the affected *set* changes, not just its size):

    - **Same budget, two independent starts: the flipped-position set is IDENTICAL — Jaccard 1.000 at all three
      budgets.** Not merely the same count; the same positions.
    - **Different budgets: the sets are NESTED, not disjoint.** 2k ⊂ 16k (shares 119 of 119) and 2k ⊂ 4k (117 of 119);
      16k ⊂ 4k (130 of 132). Jaccard 0.902 / 0.496 / 0.551.

    So the same positions are always the fragile ones and the budget decides how many of them cross the flip threshold.
    **That is one underlying instability whose magnitude packing modulates — not different shapes exciting different
    mechanisms.** It also bounds the remaining search: whatever the reduction is, it perturbs a fixed, reproducible set
    of ill-conditioned positions, which is a far easier target than a floating one.

    **Clarification against my own earlier over-correction.** Having found that the per-start raw payload hashes differ
    and the completion hashes permute, I told the user I had overstated the reproducibility. That was itself wrong on
    the point that matters: the *flipped-position set* is identical across starts (Jaccard 1.000); what permutes is the
    assignment of the eight outcomes to repeat indices. Both hold, and they are consistent — the same multiset of
    per-slot outcomes, dealt in a different order, leaves the set of positions where the eight disagree unchanged.


189. **NO CONTENT LEAKAGE between concurrent requests — and the probe that appeared to find some was answering the
    wrong question (`xtalk` + a control computed from `conc1`, 2026-09-09).** Asked because det-186 structurally cannot
    detect leakage: all eight of its concurrent repeats are the same prompt, so identical content crossing between
    requests would be invisible. The probe fired the English target alongside seven co-tenants built only from
    CJK/Cyrillic/Greek/Devanagari tokens and asked whether any co-tenant token enters the target's top-5 where it never
    does solo.

    **It reported 124 "alien intrusions", with the solo control clean (`solo_bit_identical: True`).** Decoded, the ten
    distinct tokens are `。 ， ： 是 同 那 它 短 不同 所以` — CJK punctuation *and* content words. Read naively that is
    contamination.

    **It is not.** The missing null was the same measurement with *homogeneous English* co-tenants — no CJK anywhere in
    the batch — which `conc1` had already collected:

    | arm | new top-5 entries vs solo | CJK "intrusions" | distinct CJK tokens |
    | --- | --- | --- | --- |
    | alien co-tenants | 9,252 | 124 | all ten |
    | **control: English co-tenants** | 16,605 | **178** | **the same ten** |

    The same ten tokens appear at a *higher* raw count with no CJK in the batch at all. They are the CJK tokens that sit
    just below the top-5 at particular English positions in a multilingual vocabulary — punctuation near-synonyms for
    `.` `,` `:` and high-frequency function words — so any perturbation promotes them. Per new top-5 entry the rates are
    1.34 % and 1.07 %, close, and the raw counts point the wrong way for a leakage story.

    **Both arms, with their controls, once the second arm finished:**

    | arm | co-tenants | new top-5 entries vs solo | CJK "intrusions" | rate |
    | --- | --- | --- | --- | --- |
    | MTP-3 | alien (CJK) | 9,252 | 124 | 1.34 % |
    | MTP-3 | **English only (control)** | 16,605 | **178** | **1.07 %** |
    | MTP-0 | alien (CJK) | 2,366 | 35 | 1.48 % |
    | MTP-0 | **English only (control)** | 16,907 | **232** | **1.37 %** |

    `solo_bit_identical` is True in both arms and `new_topk_entries_vs_solo` is well above zero in all four cells, so
    both pre-committed controls pass. In each arm the English-only control produces **more** CJK promotions in absolute
    count than the arm with CJK actually in the batch, and the per-new-entry rates differ by less than the difference
    between the two control cells themselves.

    **Verdict: no content leakage detected at top-5 resolution.** Stated with its limit: a leak that perturbs values
    without promoting a co-tenant token into the top 5 would not be caught by this instrument.

    **The design error is the useful part, and it is mine.** I wrote into the runner that an alien token "cannot be
    promoted into the top-5 of English prose by rounding — it would have to jump thousands of plausible continuations".
    That was an assertion, not a measurement, and it was wrong: in a multilingual model those tokens are *already* near
    the boundary. A positive signal with an unexcluded benign explanation is not evidence. **The rule this adds: a probe
    needs its null measured in the same run, not argued for in the header comment.** Had this been reported as
    contamination it would have gone to vllm#56009, which is an open cross-talk report with no reproducer — precisely
    the thread where a false positive would do the most damage.

    Third design failure of this class in two days (finding 157's gate, the payload-fingerprint extraction, this). All
    three shared a shape: a check that could return a confident answer without ever having tested what it claimed to.

    **And the worst detail: the error propagated into its own reviewer.** The wrong premise was written into the
    runner's header comment *and* copied verbatim into the watchdog prompt that was supposed to audit the result — "an
    alien token cannot be promoted into the top-5 by rounding … that outranks every other open item … draft it for
    vllm#56009". Read literally, the reviewer instructed the reviewer to report the false positive upstream. **A probe
    and its audit must not share a premise.** When writing a gate, the harvest instruction should state the *null that
    would refute it*, not restate the hypothesis with more confidence.

---

## det-190 — the top-k boundary never ties on this traffic: #55122's defect is unreachable end to end

**Run:** `tiecensus` (2026-09-09 14:06), the census patch re-inserted into `qsa_indexer._topk` on
`vllm-venv-fnmain2`, 400 instrumented calls over two prompts (`repeat`, `prose`, 13,639 chars each),
FN_MAXLEN 16384, FN_BATCH 4096, MTP 3, k = 512 blocks throughout.

A row is **ambiguous** iff `n_gt < k < n_gt + n_eq` — the kernel must choose only `k - n_gt` of the
`n_eq` entries tied at the k-th value, and *which* it picks is the arrival-order behaviour our PR
#55122 makes deterministic.

| | |
| --- | --- |
| rows seen | 87,257 |
| rows where selection is a **no-op** (`visible ≤ k`, everything taken) | 81,065 (92.9 %) |
| rows that performed a **real selection** | 6,192 (7.1 %), in 267 of 400 calls |
| rows with **any tie at the k-th value** (`n_eq > 1`) | **0** |
| rows **ambiguous** | **0** |

Two separate reasons the defect does not fire here, and they should not be conflated:

1. **On 93 % of rows there is nothing to select.** Fewer visible blocks than the budget, so the top-k
   returns everything and order is irrelevant. This is a property of the *context length*, not the kernel.
2. **On the 7 % that did select, the boundary never tied at all** — not "tied but unambiguously", *never
   tied*. `rows_with_equal_kth` is 0, so the ambiguity count is 0 for the strongest available reason.

**This corroborates det-184 from the other side.** There, `qsadet` alone moved the disagreeing-position
count 333 → 330, i.e. did essentially nothing, while all four fixes together gave 0. The census says why:
on this traffic the top-k has no tie to break, so a fix to tie-breaking cannot change the output. The
end-to-end reproducibility we report is carried by the other three (`detfin`, `cachekey`, `plefix`).

**What this does *not* say.** It does not say the kernel is correct — the standalone test still shows
arrival-order dependence under tie-heavy synthetic input, and #53287's reasoning stands. It says the
defect is **unreachable on this workload**, bounded by: 16 k context, two prompts, 6,192 selecting rows.
Longer contexts push a larger fraction of rows past `visible > k` and would sample the boundary far more;
a tie rate indistinguishable from zero at 6 × 10³ samples is not zero at 10⁶. Exact ties between float32
logits coming out of a real GEMM are simply rare.

**Consequence for the PR.** #55122 should be presented as *correctness of the kernel under ties*, not as
a fix that buys end-to-end determinism on production traffic — we have now measured that it does not,
here. Saying so ourselves is cheaper than a reviewer finding it. Draft: `notes/upstream/comment-55122-tie-census.md`.

**Data:** `/opt/llm/tiecensus.jsonl` (400 records), runner `tiecensus.sh`, patch `tiecensus_patch.py`.

---

## det-191 — the indexer SCORES are nondeterministic, not the selection: rybruscoe's discriminator answered

**Run:** `scorediv` (2026-09-09 14:25–14:55), two arms on `vllm-venv-fnmain2`, 8 byte-identical greedy
requests each (`p5960`, 34,868 chars), MTP 3, FN_MAXLEN 16384, FN_BATCH 4096. An env-gated probe at the
end of `qsa_indexer._topk` hashed, per call, the **input scores** (`shash`) and the **kernel's selected
block indices** (`ihash`) over the first 32 selecting rows.

This is the discriminator @rybruscoe proposed on vllm#54521, which det-190 could not settle: since the
boundary never ties exactly, a differing selection between two identical requests must be either
(1) the scores differing upstream, or (2) a genuine ordering bug in the top-k.

**Result — 13 aligned prefill calls compared across 7 identical warm requests per arm:**

| | scores+selection identical | **scores differ**, selection differs | scores differ, selection same | **scores identical, selection differs** |
| --- | --- | --- | --- | --- |
| `stock` (no fixes) | 0 | **13** | 0 | **0** |
| `all4` (all four fixes) | **13** | 0 | 0 | **0** |

**The answer is (1).** In stock, every comparable call has *different input scores*, and not one call has
identical scores with a different selection. The top-k kernel is where the divergence becomes visible,
not where it originates — so **#55122 is not the root cause of end-to-end nondeterminism**, exactly as
det-184 implied (qsadet alone: 333 → 330) and as we already told upstream. `all4` is bit-identical on
both hashes, which is an independent confirmation of det-184 from a different instrument.

Boundary gaps (kth − (k+1)th score), stock: min **4.58e-05**, median **3.20e-04**. So a score
perturbation of order 1e-4 is enough to reorder the boundary. **We did not measure the perturbation
magnitude** — the probe stores hashes, not values — so this is the scale at which it *could* flip, not a
demonstration that it does. rybruscoe's "near-tie band" prediction is *consistent* with this and not yet
confirmed; confirming it needs the scores logged as values.

### The run was VOID as designed, and the first analysis was wrong

Both of my pre-registered void criteria fired: the stock arm reported **0 comparable calls**, and the
`all4` control reported **176 calls with differing scores** when it should have been identical. Neither
was a property of the model.

**Defect 1 — the cold request.** Segment 0 carries extra warm-up calls (148 in `all4`, 475 in `stock`),
so comparing call *i* across segments misaligned every comparison that included it. The whole "176"
figure was that artifact; with segment 0 dropped, `all4`'s 7 remaining segments are 405 calls each and
align exactly.

**Defect 2 — decode calls cannot align in the stock arm, by construction.** When outputs diverge, MTP
acceptance differs, so the *number of decode steps* differs per request (stock segment lengths: 478,
510, 390, 345, 405, 345, 480, 510). Call-by-call alignment is impossible there — and that divergence in
call *count* is itself a symptom of the thing being measured.

**The fix, applied to the data already collected rather than by re-running:** drop the cold request, and
compare **prefill calls only** (`rows > 64`). Prefill count and shape depend on the prompt alone, so they
cannot drift with generated tokens. Both arms then give 13 comparable calls per request with identical
row counts, and the table above is that comparison. Analyser: `/opt/llm/runners/sd/sdre.py`, summary in
`notes/data/scorediv-summary.txt`.

**Bound:** prefill only. Decode-time behaviour is untested here, and a re-run wanting decode coverage
must key calls by content rather than by index.

---

## det-179 — kernel-det v2.4 has a context ceiling on GB10 at ~93.6k tokens

**2026-09-10, found by accident during the stage-1 re-capture (`fx-lhcap`).**

With `VLLM_QSA_DET_TOPK=1`, a 95,239-token prefill dies at:

```
RuntimeError: launch_persistent_topk, /opt/llm/runners/kdet_build/topk_det.cu:117,
persistent_topk_det: dynamic smem 97568 exceeds 97120 (optin 101376 - static 4256)
```

It had written **94,080 of 95,239 rows** first, so the failure is at ~94k positions and the request
overruns the limit by **448 bytes**. If the dynamic shared-memory request is linear in position, the
ceiling is **~93,648 tokens**.

GB10 is the constraint: 102,400 B of shared memory per SM, 101,376 B opt-in, minus 4,256 B static
leaves 97,120 B for the kernel's dynamic request.

**Why this has never surfaced:** `FN_MAXLEN` defaults to **8,192** in `serve-fnmain.sh`, and every
A/B this year has run at 8k–32k. Nothing in normal operation approaches 93k, and the stratified
sampler only hit it because it feeds the corpus's longest instance first.

**Consequence to keep in mind:** the deterministic top-k overlay cannot be used at near-100k context
on this box. Prod is unaffected at its current context, but any future long-context work has to
choose between the overlay and the context.

The capture itself does not need determinism — it collects activation distributions, not bit-exact
outputs — so stage 1 re-ran with `FN_DET_TOPK=0`, which is the documented stock arm.

**Second defect, same hour, worth its own line:** `lhcap.sh` had `VLLM_QSA_DET_TOPK=1` **hardcoded**,
so passing `FN_DET_TOPK=0` as a systemd `Environment=` property did nothing and the relaunch would
have failed identically ten minutes later. Caught by checking the runner rather than trusting the
launch. Every runner that takes an arm flag must read it as `${FN_X:-default}` — verified after
launch by reading `/proc/<pid>/environ`, which is now the habit.

> **Numbering note (2026-09-11):** finding numbers are **not monotonic in file order** — det-190 and
> det-191 sit above det-179/det-180. Take the next number as `max + 1` over the whole file, never
> "the last heading + 1". det-179 and det-180 were both numbered the wrong way; they collide with
> nothing, so they stand.

## det-180 — the sm_121 W4A16/W4A4 kernel mis-selection is REAL upstream and INERT for us, 2026-09-11

**Upstream bug confirmed on our hardware** (issue #55397, fix #55405). On sm_121, walking
`_POSSIBLE_NVFP4_KERNELS[CUDA]` in order with `use_a16=False`:

| # | kernel | `is_supported()` on sm_121 |
| --- | --- | --- |
| 1 | `FlashInferCuteDslNvFp4LinearKernel` (**W4A4**) | **False** — "requires sm_10x" |
| 2 | `FlashInferCuteDslNvFp4W4A16LinearKernel` (**W4A16**) | True — accepts sm_100 **or sm_12x** |
| 3–8 | FlashInferCutlass / B12x / Cutlass / Marlin / Trtllm / Cudnn (W4A4) | True, never reached |

The CuTe-DSL W4A4 kernel excludes sm_12x while its W4A16 sibling admits it, and the W4A16 kernel's
`can_implement()` returns `(True, None)` unconditionally — it accepts a W4A4 config. Verified by
calling the selector, not by reading it:

```
init_nvfp4_linear_kernel(use_a16=False) -> FlashInferCuteDslNvFp4W4A16LinearKernel
init_nvfp4_linear_kernel(use_a16=True)  -> MarlinNvFp4LinearKernel
```

`apply_weights` takes BF16 `x` and never quantizes activations, so a W4A4 checkpoint's `input_scale`
would be ignored for any dense linear that reached this path.

### It does not reach that path on our checkpoint

Counting modules with a `weight_scale` sibling in the base index:

| NVFP4-quantized | count |
| --- | --- |
| `layers.N.mlp.experts.N.{gate,up,down}_proj` | 73,728 (48 × 512 × 3) |
| `layers.1.ple.ple_embedding.ngram_embedding` | 1 |
| everything else | **0** |

`exclude_modules` removes `*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`, `*.mlp.shared_expert.*`,
`*hyper_connection*`, `*.ple.*`, `lm_head`, `mtp.*`, both embed tables. There is **no quantized dense
Linear in this model**, so `init_nvfp4_linear_kernel` is never consulted for it; the routed experts go
through the fused MoE path and the PLE entry is an embedding, not a Linear.

**Independent confirmation that the expert path really is W4A4:** the `input_scale` contract bug fixed
2026-09-10 corrupted Thai combining marks. A weight-only kernel ignores `input_scale`, so it could not
have produced that corruption. The experts consume activation scales.

### Consequence for [[w4a16-vs-w4a4-measured]]

The queued doubt was: *if the W4A4 arm's dense linears ran through a 16-bit-activation kernel, the
0.42 pp fidelity gap is suspect.* **That doubt is retired** — the W4A4 arm has no quantized dense
linears to mis-route. The gap came from the MoE path, which the TODO already noted is unaffected.
The finding stands as measured.

**Action for Flash-Next: none.** #55405 changes a selection this checkpoint never makes.

### ~~CORRECTION — the dense 27B IS affected~~ **WITHDRAWN 2026-09-11 07:5x, see below**

The section that follows inferred "no `input_scale` tensors ⇒ W4A16 ⇒ `use_a16=True` ⇒ forced
Marlin". **Every step after the first is wrong, and the first is wrong too.** Kept in full because
the reasoning is instructive; read the withdrawal at the end before using any of it.

Scope error: the paragraph above is true of **Flash-Next**, not of our fleet. The dense
`qwen38-27b-nvfp4` is the opposite case and it is the model we prefer for long work
([[dense-27b-preferred-for-long-work]]).

| checkpoint | quantized modules | with `input_scale` | quantized dense Linears |
| --- | --- | --- | --- |
| `qwen38-flash-next-nvfp4` | 73,729 | 73,728 | **1** (a PLE embedding) |
| `qwen38-27b-nvfp4` | 401 | **0** | **401** |

The 27B's 401 are `mlp.{gate,up,down}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` and
`self_attn.{q,k,v,o}_proj` — every one a dense Linear, and **none carries an `input_scale`**, so it is
a W4A16 checkpoint by construction and `use_a16=True`.

That takes a **different branch** from the registry walk traced above:

```python
elif linear_backend == "auto" and use_a16:
    if compute_capability in (100, 103) and cutedsl_ok:
        force_kernel = FlashInferCuteDslNvFp4W4A16LinearKernel
    else:
        force_kernel = MarlinNvFp4LinearKernel     # sm_121 lands here
```

sm_121 is not in `(100, 103)`, so all 401 layers are forced onto **Marlin** — while
`FlashInferCuteDslNvFp4W4A16LinearKernel.is_supported()` returns **True** on sm_12x (measured, table
above). The hard-coded SM list contradicts the kernel's own support gate. Confirmed by calling the
selector: `init_nvfp4_linear_kernel(use_a16=True) -> MarlinNvFp4LinearKernel`.

**Cheap test, not yet run.** The force only applies when `linear_backend == "auto"`. Setting
`--linear-backend flashinfer_cutedsl` should skip the force, filter the a16 candidates
(`CuteDslW4A16`, `Marlin`, `Humming`) to that backend, and select CuteDSL W4A16. One flag, no patch.
**Unverified** — `_get_linear_backend()` needs a live vLLM config, so this was read, not executed.

**Queued:** A/B the 27B on sm_121, `--linear-backend auto` (Marlin) vs `flashinfer_cutedsl`, c=1 decode
+ TTFT, three starts. If CuteDSL wins, #55405's territory matters to us after all — on the 27B, not on
Flash-Next — and the SM list is worth reporting upstream with a GB10 number attached.

**What does NOT change:** the [[w4a16-vs-w4a4-measured]] comparison was run on Flash-Next, where both
arms have W4A4 experts and no quantized dense Linears. Its 0.42 pp gap is still not explained by this.

### WITHDRAWAL — the 27B is W4A4 with *dynamic* activations, and it runs CUTLASS

Checked against the actual run logs with `kernelroster.py` (below) instead of by reading dispatch
code. Thirteen 27B server logs, all on `/opt/llm/models/qwen38-27b-nvfp4`, all say:

```
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
```

**Not Marlin.** Zero occurrences of "marlin" in any of them. The premise was the error:

| I claimed | Actually |
| --- | --- |
| 0 `input_scale` tensors ⇒ weight-only (W4A16) | **dynamic** activation quant stores no `input_scale`; the scale is computed per token at runtime |
| takes the ModelOpt `use_a16` branch | the checkpoint is **compressed-tensors**, `quantization=compressed-tensors` in the log — a different scheme class with its own dispatch |
| forced to Marlin on sm_121 | selects `FlashInferCutlassNvFp4LinearKernel`, a W4A4 kernel |

`config.json` settles it — `config_groups.group_1`, targeting `re:.*mlp\.(gate|up|down)_proj$`:
`weights num_bits=4, dynamic=False, group_size=16` and
**`input_activations num_bits=4, dynamic=local, group_size=16`**. W4A4. (`group_0` — attention,
linear_attn, `lm_head`, the last eight layers' MLP — is W8A8 FP8 dynamic.)

Also: those logs are **vLLM 0.27.1**; the dispatch code I traced is the current **0.28.1rc1** venv.
Reading today's source to explain a measurement taken on an older build is its own mistake.

**Consequence: the doubt cast on the published NVFP4 Periodic Table is withdrawn in full.** I claimed
its two `W4A16 · Marlin` columns (10 measured cells) might be misattributing an SM-list default to the
quantization scheme. No evidence supports that: the one cell whose log I can read ran
`W4A4 → CUTLASS`, which is exactly what the map's column header says. The map's rule
("W4A4 forces CUTLASS, W4A16 forces Marlin") is unrefuted and, for the W4A4 half, now directly
confirmed from a run log. **Nothing on that page needs changing.**

What survives: the sm_121 forced-Marlin branch is real in 0.28.1's **ModelOpt** path for a genuinely
weight-only checkpoint. We do not currently serve one, so it remains untriggered here — the same
verdict as for Flash-Next, reached for a different reason.

## det-192 — PLE mmap is a 3-file port, but it REQUIRES `--enforce-eager`, so det-158 gates it

Desk review of `patches/vllm-complete.patch` in
[Radar105/qwen38-flash-next-nvfp4-spark](https://github.com/Radar105/qwen38-flash-next-nvfp4-spark)
(sha256 `e05d11568ddd96f0…`, 1,755 lines, 23 files). No venv was touched.

**The mmap reader does not need the other 20 files.** It is three pieces:

| piece | where |
| --- | --- |
| `MmapPLEEmbedding` (new file) | `vllm/models/qwen4_exp/nvidia/ple_mmap.py` |
| `VLLM_QWEN4_PLE_MMAP` flag | `vllm/envs.py` |
| dispatch before `PLEVocabParallelEmbedding` | `vllm/models/qwen4_exp/nvidia/ple_layer.py` |

The other 20 files are five stacked upstream PRs plus their tests; the TODO's worry about having to
take the whole stack is unfounded.

### Three hard gates, and one of them is the problem

1. **TP1 only** — we are TP1. Fine.
2. **F8_E4M3 checkpoint rows only** — our PLE is FP8. Fine.
3. **`--enforce-eager` required** — checked twice: at config time, and inside `forward()` via
   `torch.cuda.is_current_stream_capturing()`.

### Why it needs eager: the lookup runs on the CPU

`forward()` does `indices.to(device="cpu")`, `np.unique` to dedup, numpy fancy-indexing into the
`np.memmap` shards, then `.to(indices.device)`. That is a device→host sync plus page-faulted host
reads plus a host→device copy **per PLE call** — impossible inside a captured graph, hence the gate.
It is not "map the table into GPU memory"; it is a host-side gather with a dedup in front of it.

### Consequence: det-158 is now a prerequisite, not a sibling

We serve with `cudagraph_mode: PIECEWISE`, which `--enforce-eager` would disable. So the mmap path is
unusable for us **unless cudagraphs are already doing nothing here** — exactly what det-158 suspects
(`0.0 GiB for CUDAGraph memory`, zero `Capturing CUDA graphs` lines in both arms of the capture-width
A/B). Radar105 running `--enforce-eager` in their own production is independent support for that.

**Reordered:** det-158 (`FN_CG_MODE=NONE` arm) must land before any mmap port. If cudagraphs are inert,
eager is free and the port is worth doing; if they are live, the port costs whatever they are worth and
the ngram question needs a different route.

**Not established:** whether the mmap path avoids the V1 executor rejection that
`VLLM_PLE_CPU_OFFLOAD` triggers (det-160). It uses no offload worker, so it plausibly does, but that
is a claim about a code path I have not run. It also substitutes its own constraint, so "ngram is
unblocked" does not follow from "the offload conflict is gone".

### The published 27B map depends on this — 10 measured cells

The [NVFP4 Periodic Table](https://claude.ai/code/artifact/1c0eb53f-9daf-406e-ad6f-149ac0ac161f)
is the 27B map, and it states the kernel rule as a property of the format:

> "The kernel is a property of the column, not of a cell: W4A4 forces CUTLASS, W4A16 forces Marlin.
> … Where the kernel does show is prefill: 6.4 s against 8.4 s TTFT."

Its four grid columns are labelled accordingly, with the penalty attached to the header:

| column | header claim | measured cells |
| --- | --- | --- |
| W4A16 · Marlin, dynamic act. | slow prefill · TTFT ~10 s | 7 |
| W4A16 · Marlin, static act. | slow prefill · TTFT ~8.5 s | 3 |
| W4A4 · CUTLASS, dynamic act. | fast prefill · TTFT ~6.3 s | 5 |
| W4A4 · CUTLASS, static act. | fastest prefill · 4.6 s | 4 |

**"W4A16 forces Marlin" is true on sm_121 and is not a property of the format.** It is the
`compute_capability in (100, 103)` literal in `init_nvfp4_linear_kernel`: on an sm_100/103 box the
same W4A16 checkpoint takes CuteDSL W4A16 instead. The kernel's own `is_supported()` admits sm_12x,
so the literal and the gate disagree, and GB10 falls to Marlin on the `else`.

**What this does and does not put in doubt.** The *measurements* stand — those cells really were
Marlin, and the TTFT figures really were observed. What is not established is the map's *causal
attribution*: a reader takes "W4A16 ⇒ slow prefill" as a fact about the quantization scheme, when on
this box it may be a fact about one hard-coded SM list. If `--linear-backend flashinfer_cutedsl` lifts
it, the 2–4 s spread between the map's halves is partly a vLLM default, not a scheme cost, and the
column headers need rewording.

**Do not edit the page yet.** The A/B is unmeasured and the page is published and shared; the
`nvfp4-table` skill owns it and its sources live in `bench/nvfp4-table/`. Sequence: run the A/B
(`--linear-backend auto` vs `flashinfer_cutedsl`, W4A16 cell, TTFT + c=1 decode, three starts), then
go through the skill.

## det-193 — prod captures NO cudagraphs, and it is not the capture sizes, 2026-09-11 08:33

**det-158 answered, and the A/B it specified was void before it started.** Seven server starts across
three configurations, all on `qwen38-flash-next-nvfp4`, `enforce_eager=False`, vLLM
`0.28.1rc1.dev401+g8340fe1bb`:

| log | mode | capture sizes | max | CUDAGraph memory | `Capturing CUDA graphs` lines |
| --- | --- | --- | --- | --- | --- |
| cgab-piece1/2/3 | PIECEWISE | [1,2,4,8] | 8 | **0.0 GiB** | **0** |
| cgab-full1/2/3 | FULL_DECODE_ONLY | [1,2,4,8] | 8 | **0.0 GiB** | **0** |
| **cgsize-wide** | PIECEWISE | **[1,2,4,8,16,32,64]** | **64** | **0.0 GiB** | **0** |

### Why det-158's own design was void

It specified `PIECEWISE` vs `NONE`. But PIECEWISE already captures nothing, so `NONE` would have been
a third non-capturing configuration and the two arms would have been the same behaviour under
different names. The six `cgab` logs had been on disk since 2026-09-10 and said so; nobody had read
them for this. **Fifth void run identified — and the first caught before spending the starts.**

The tool that caught it is `kernelroster.py` ([[kernel-roster-from-logs]]), written the same morning
for an unrelated question.

### The one new start, and what it rules out

`fx-cgsize` with `FN_CG_SIZES=[1,2,4,8,16,32,64]`. **The knob reached the cell** — the engine built
with `max_cudagraph_capture_size: 64`, up from 8 — and the outcome did not move. So this is a valid
negative, not a void run: capture is not being skipped because the sizes are too small.

The motivating hypothesis is dead: MTP n=3 makes each decode step 1+3 tokens per sequence, so a real
batch never lands on 1/2/4/8 — but widening to 64 covers every shape this server can produce
(`max_num_seqs 4`) and still nothing captures.

### What this settles

- **det-136's null has a second explanation.** Both its arms had capture disabled in fact, whatever
  they asked for.
- **The capture-width A/B is void**, not merely null, and must not be re-run as designed.
- **MiaAI #19 is discharged**: the answer to "does it capture" is no, with seven starts behind it.
- **det-192 (PLE mmap) is UNGATED.** Its blocker was `--enforce-eager` conflicting with our PIECEWISE
  config. If PIECEWISE captures nothing, eager costs nothing, and the 3-file port is worth doing.

### What is NOT established

**Why** capture is skipped. Ruled out: capture size, cudagraph mode, `enforce_eager`. Still open, and
both are source-reading hypotheses that this session's record says to distrust until a log confirms
them:

1. the speculative/MTP path suppressing capture, or
2. `splitting_ops` listing nearly every op this model uses — `qwen4_exp_qsa_with_output`,
   `qwen_gdn_attention_core`, `qwen4_exp_ple_short_conv`, `linear_attention`, `mamba_mixer2` — which
   under PIECEWISE would leave no capturable region between splits.

A cheap discriminator exists: serve **without** `speculative_config` and see whether capture appears.
One start, one differing cell. Queued, not run.

## det-194 — speculation is NOT why capture is skipped; three of four candidates are now dead

The discriminator det-193 queued. One start, `FN_MTP=0`, everything else at det-193 settings. The
differing cell was reached — the log carries `speculative_config=None`, so MTP was genuinely off —
and the outcome did not move.

Eight starts, four configurations, every one `0.0 GiB` with zero `Capturing CUDA graphs` lines:

| log | mode | capture sizes | speculation | CUDAGraph | capture lines |
| --- | --- | --- | --- | --- | --- |
| cgab-piece1/2/3 | PIECEWISE | [1,2,4,8] | mtp n=3 | 0.0 GiB | 0 |
| cgab-full1/2/3 | FULL_DECODE_ONLY | [1,2,4,8] | mtp n=3 | 0.0 GiB | 0 |
| cgsize-wide | PIECEWISE | [1,2,4,8,16,32,64] | mtp n=3 | 0.0 GiB | 0 |
| **cgnospec** | PIECEWISE | [1,2,4,8] | **OFF** | **0.0 GiB** | **0** |

### Ruled out

1. `enforce_eager` — `False` in all eight.
2. **cudagraph mode** — PIECEWISE and FULL_DECODE_ONLY behave identically.
3. **capture sizes** — `max_cudagraph_capture_size` 8 → 64 changes nothing.
4. **speculation / MTP** — off changes nothing. *(det-194, this finding.)*

### What is left

The `splitting_ops` hypothesis: under PIECEWISE, vLLM captures the regions *between* splitting ops,
and this model's forward is almost entirely ops on that list — `qwen4_exp_qsa_with_output`,
`qwen_gdn_attention_core`, `qwen4_exp_ple_short_conv`, `qwen4_exp_compute_ple_ngram_ids`,
`linear_attention`, `mamba_mixer2`, `short_conv`, `unified_kv_cache_update`. If every op is a split
point there is no region left to capture, and zero is the correct output rather than a bug.

That would also explain FULL_DECODE_ONLY behaving the same way: this is a hybrid GDN/QSA model whose
decode path runs through stateful custom ops, so a whole-graph capture has nothing it can legally
snapshot either.

**This remains a source-reading hypothesis and is deliberately not asserted.** The clean test is a
model *without* those custom ops on the same venv and GPU — a plain dense transformer — where a
non-zero CUDAGraph figure would prove the stack can capture at all and isolate the cause to this
model family. One start, and it needs a second checkpoint rather than a flag.

### Practical consequence, which does not wait on the cause

Cudagraphs are **inert on this model**, across every knob we can reach. So:

- `--enforce-eager` costs nothing here, and **det-192's PLE mmap port is ungated** — that is the
  actionable item, and it does not depend on knowing why.
- Any future A/B that varies a cudagraph knob on Flash-Next is void before it starts. Check
  `kernelroster.py` output first ([[kernel-roster-from-logs]]).
- det-136's null and the capture-width null both have this as a sufficient explanation.
