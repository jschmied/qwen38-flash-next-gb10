# Queued: re-measure the MTP cells on the fixed prod stack, then update the published page

Queued 2026-09-06 on the user's instruction ("queue re-measurement and page update for later on the
prod stack with all fixes"). Nothing here is started. The published Quant Map keeps the affected
cells until this runs — that is the accepted trade, not an oversight.

## Why every MTP figure on the page is owed a re-run

| defect | what it does to an MTP number | fixed |
| --- | --- | --- |
| PLE conv reads a strided `state_indices[:,0]` with unit stride | target output **corrupted** on ~50 % of simultaneous prefill pairs with MTP on; acceptance ~9 % for the affected request. Every **c≥2** MTP cell measured generation that was partly garbage. | vllm#55375, merged 2026-09-05; **in the prod venv** (`qwen4_exp/nvidia/ops/ple.py`, `state_idx_stride`) |
| PLE offload semaphore one step behind under cudagraphs | every forward consumed the previous step's PLE rows since the #53899 port (2026-09-03) | our fix, in prod since 2026-09-06 17:4x |
| `disable_eagle_block_drop` unset | each warm turn re-prefills one 1,600-token block it need not | default on the `FN_MTP` path since 2026-09-06 |
| MTP restart instability (not a defect — a property) | worst within-config spread **1.83×** across server starts (MTP2 47.8→77.5 ms/tok), against 1.10× no-spec and 1.09× ngram. Any MTP timing from a single start is uncallable. | not fixable; it is a protocol requirement |

## Protocol (non-negotiable, all earned)

- **Three starts minimum per MTP cell**, report the **range**, not a mean. Gaps under 2× are
  unresolved unless the ranges separate. A 2-sample agreement proves nothing on this path.
- **c≥2 probes use a barrier** (`stagger.py` / `decode_cell2.py`) — the old sequential-thread probe is
  what hid the corruption for weeks.
- Accept-length pinned at the maximum is a **corruption signature**, not health. Empty content is
  detected by `finish_reason: "length"`, never by counting characters.
- Prefix-cache hits come from `prefix_cache_hits_total` deltas; `usage.cached_tokens` is inert.
- Interleave arms across starts rather than running an arm three times in a row.
- Per-arm `FN_CACHE_ROOT` and a purge: the compile-cache key omits `num_speculative_tokens` and
  env-gated branches, so an n-sweep otherwise shares one graph.

## Cells, in the order they pay

**1 — the page's headline agent-loop claim (~18 starts).** Fixed-work loop, `ignore_eos`, 8 × 130
tokens: no-spec / ngram n=4 / ngram_gpu n=4 / MTP k=2 / MTP k=3, three starts each, MTP arms with
the flag on. Take the c=1 decode ladder inside the same starts so the ladder and the loop come from
one stack. Expected to confirm, with a better basis, the two results that already survived the
restart finding: k=2 non-overlapping with no-spec, and n=3/4/5/6 reaching a floor k=2 never reaches.

**2 — the c≥2 cells that measured corruption (9 starts).** c=16 aggregate and TTFT for off / k=2 /
k=3, barrier-staggered. These are the page's `99.1 / 100.5`, `6.79 / 7.29 s`, the "−30 % TTFT"
claim and the c=16 column of the compose table.

**3 — the depth curve (12 starts).** Legal n only: 2, 3, 4, 9 (5..8 need the widened ring). Three
starts each, ranges. This replaces both the published curve and its withdrawn debug-instrumentation
explanation.

**4 — the ~68-token threshold, re-derived with the flag on (6 starts).** Forced 30- and 400-token
turns, MTP k=3 against off. The block term the threshold is built on is no longer paid.

**5 — FP8 KV c=1 MTP arms (6 starts, lowest).** The published `36.28` vs `37.22`.

≈ 51 starts, ~8 min each → an overnight job. `mtpnodrop2` (running 2026-09-06 21:14, ON/OFF × 3
starts, `agentloop2.py`) already covers the flag delta; do not repeat that cell.

## The page edits this unblocks

`bench/flashnext-quants/flashnext-quants.html` in the setup guide, artifact
`3534a530-5e94-4ce2-abac-f1c70ee204e3`:

1. Replace the speculation table's c=16 cells and the compose table's c=16 column with re-measured
   ranges; drop or restate "cuts TTFT by 30 %".
2. Replace the depth-curve callout: the debug-instrumentation attribution is **withdrawn** (a 2.2 %
   → 2.2× claim from two n=1 samples, inside the 1.83× spread). State the restart spread, the two
   results that survived it, and the new curve.
3. Restate the noise floor: 6.9 % is a **decode** floor derived from k=2 runs and does not cover MTP
   arms across starts.
4. Close the `~26 %` caveat added 2026-09-06 with the measured number, and re-derive the ~68-token
   threshold or withdraw it.
5. Mark the concurrency-ceiling table's c=16/c=32 rows for whichever arms had MTP on.
