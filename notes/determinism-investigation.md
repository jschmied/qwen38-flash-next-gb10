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
    leaves slots >= visible untouched, so they hold `torch.empty` garbage; sorting the whole row
    moved garbage below the real indices, where the expansion kernel reads them. `torch.topk`'s
    padding is all >= visible, so it was immune. T5/T6 in `topk_boundary.py` will confirm.

    **Which earlier arms are affected**: every underfilled probe that ran `persistent_topk` with
    the sort — `NOPFX_a/b/c`, `BISECT`, `BISECT2` (the `'The'` / -1.x band). NOT affected: arms
    with `torch.topk` (`P_ctl`, `P_nospec`, `F_noprefix`, `G_*`), and NOT the agent-loop /
    `degrade` arms (an 8k+ prompt gives ~2000 visible blocks > k=512, so no padding exists).

    **Which conclusions survive**: all the categorical ones. "1 distinct of 3" is still
    deterministic and "3 of 3" still diverges whatever the token; the layer-1 bisection is still
    a divergence. What is retracted is any *value* quoted from a sort-affected arm — the -1.51
    logprob, the `'The'` token, and finding 2's "top-k changes the answer", which is withdrawn.

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
