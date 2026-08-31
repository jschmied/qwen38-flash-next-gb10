# Which drafter for agent work: the decode ranking is exactly inverted

Measured 2026-08-31 on the fixed harness (`tools/agentloop.py`: `ignore_eos`, work pinned at
8 × 130 tokens, `ms/tok` reported, unequal work refused). `fp8head`, batch 4096,
`max-model-len 32768`, c=1, NIAH-gated.

| config | **ms/tok (agent loop)** | decode @4k | decode floor¹ | overhead above floor | KV pool |
| --- | ---: | ---: | ---: | ---: | ---: |
| **ngram n=4** | **33.3** (33.26 / 33.39) | 30.5–31.9 | ~32 | **1.3** | −17% |
| no speculation | 43.37 | 27.0 | 37.0 | 6.4 | — |
| MTP k=2 | **47.75** | **38.6–40.6** | ~25 | **22.8** | −36% |

¹ `1000 / decode_tps`, i.e. the per-token time if a turn were pure decode with no other cost.

**MTP k=2 has the best decode on this box and the worst agent loop — worse than no speculation at
all.** ngram n=4 has the third-best decode and the best agent loop by a wide margin. Selecting a
drafter on tok/s picks exactly the wrong one.

## Where MTP's decode advantage goes

The overhead column. MTP k=2 spends **~23 ms/tok** beyond its decode floor against no-spec's 6.4
and ngram's 1.3. It pays for the drafter twice on a conversation that reuses its prefix:

- **the `use_eagle` prefix-cache back-off** — `scheduler.py` drops one cacheable block per request
  when `use_eagle` is set, and MTP sets it. Every turn re-prefills what it just cached
  ([[mtp-vs-prefix-cache]]).
- **the KV pool** — MTP doubles the attention block size (800 → 1600) to keep the attention page
  ≥ the mamba page, costing 36% of the pool ([[speculation-costs-kv-pool]]).

`ngram` is in neither trap: not in `EagleModelTypes`, so no back-off, and its block size barely
moves (848 vs 800), so it keeps its cache and most of its pool.

## Recommendation

- **Agent / multi-turn work → `ngram`, n=4.** 23% faster per token than no speculation, 30% faster
  than MTP k=2, at half MTP's KV cost.
- **Single-shot long generation → MTP k=2.** Its +45-50% decode is real and sustains at depth
  (41 tok/s at 28k); with no prefix to reuse, the back-off costs nothing.
- **k=2 is confirmed the MTP optimum.** It beats n=6 on *both* decode and TTFT, so the deep band is
  not merely unnecessary, it is worse ([[mtp-vs-prefix-cache]]).

## The advantage grows on edit-shaped work

Every arm above uses prose-like prompts. Re-run with an edit-shaped loop — each turn returns the
same function with one identifier changed, so most output tokens already exist in the context:

| config | generic ms/tok | edit-shaped ms/tok | gain from repetition |
| --- | ---: | ---: | ---: |
| **ngram_gpu n=4** | 33.82 | **26.18** | **−23%** |
| ngram n=4 | 33.3 | 29.27 | −12% |
| MTP k=2 | 47.75 | 36.85 | −23% |
| no speculation | 43.5 | 43.80 | 0% |

**`ngram_gpu` is the fastest configuration measured on this box for agent work** — 26.18 ms/tok,
40% better than no speculation and 29% better than MTP k=2.

⚠️ **On prose, `ngram_gpu` looked like a clean null** (33.82 against ngram's 33.3) and was written
up here as one: "CPU-side matching was never the bottleneck, so there was nothing to reclaim." That
was true of prose and false in general. With edit-shaped context there are far more candidate
matches to search, so finding a draft *does* cost something — and that is exactly the cost moving to
the GPU removes. A single-workload benchmark would have filed the fastest config as "no benefit".

**The baseline does not move** (43.80 vs 43.5), which isolates the effect: without a drafter,
repetitive output is no faster than prose, because decode speed does not care what the tokens are.
Every drafter gains; only the baseline is flat.

**MTP k=2's anomaly is prose-specific.** On edit-shaped work it behaves sanely at 36.85 — better
than baseline, as a drafter should be. It is only on generic prompts that it lands *worse than no
speculation at all* (47.75 vs 43.5), which remains unexplained and is under replication.

Real agent turns — quoting a file back, re-emitting a function with a small change — sit closer to
this end than to prose, so **−33% is the more representative figure** for agent use.

It is much smaller than the field's llama.cpp numbers (3.8× on copy-heavy Python). Two plausible
reasons: our probe renames one identifier rather than echoing verbatim, and vLLM's `ngram` may
draft less aggressively than the `ngram-mod` variant measured there. Both are testable; neither is
tested.

## Caveats

- ngram n=2 arms, MTP k=2 n=1 clean (its first arm was contaminated by a probe run against the same
  server — turn 8 took 34.4 s against a 4-9 s baseline; excluding it that arm implied ~47.4 ms/tok,
  which is what the clean arm measured).
- **The arms do not share an allocation unit** — block sizes 800 / 848 / 1600 — and block size is
  the prefix-cache hash unit, so some of the gap is cache geometry rather than drafter quality.
- The edit-shaped comparison is n=1 per config; the generic one is n=2-3.
- This is the third independent confirmation of [[agentic-speed-is-ttft-bound]] on this hardware,
  and the first on a metric that survives replication.
