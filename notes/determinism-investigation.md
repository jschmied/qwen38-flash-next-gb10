# Determinism investigation — state of play (2026-09-01)

Index for the night's work. Detail lives in the linked notes; this is what is established, at the
confidence each item deserves, plus what was refuted along the way.

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
