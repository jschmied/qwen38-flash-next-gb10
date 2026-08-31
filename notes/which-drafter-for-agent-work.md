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

## Caveats

- ngram n=2 arms, MTP k=2 n=1 clean (its first arm was contaminated by a probe run against the same
  server — turn 8 took 34.4 s against a 4-9 s baseline; excluding it that arm implied ~47.4 ms/tok,
  which is what the clean arm measured).
- **The arms do not share an allocation unit** — block sizes 800 / 848 / 1600 — and block size is
  the prefix-cache hash unit, so some of the gap is cache geometry rather than drafter quality.
- All prompts are prose-shaped. `ngram` feeds on repetition, and an edit-shaped workload should
  favour it further; the field measures 88.5 vs 27.8 tok/s on this box from task shape alone. An
  edit-shaped comparison is queued.
- This is the third independent confirmation of [[agentic-speed-is-ttft-bound]] on this hardware,
  and the first on a metric that survives replication.
