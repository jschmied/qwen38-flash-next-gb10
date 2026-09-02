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

5. **Excluded as causes:** speculation (`P_nospec`, `G_eager_nospec`), CUDA-graph replay
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
