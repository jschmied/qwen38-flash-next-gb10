DRAFT — the user's go covers opening it ("yes 1..3", 2026-09-26). Numbers from plefill2b (speed-of-light §4z). vllm-project/vllm new PR, head jschmied:pr/ple-cold-fill (stacked on #58439).

Title: [Qwen4Exp] Checkpoint-mapped PLE: read a decode step's cold pages with readahead

## Purpose

Follow-up to #58439 (stacked on it; the diff to review is the last commit).

With `checkpoint_mapped`, a freshly started server holds almost none of the PLE table in the page cache: loading
the weights streams them through it and evicts the table (0.03 of 47.7 GiB resident at ready on DGX Spark). A
decode step's rows are known only once the previous step has sampled, so the host prefetch has no lead: the GPU
lookup reaches the step's ~57 missing pages at the same time. Today the prefetch touches them serially in one
thread (the `< 4096 rows` path), and the GPU faults them itself, one at a time (~150 µs each).

This PR fills decode-sized row sets with `MADV_WILLNEED` for every page, which queues readahead and returns at once
so the SSD serves the pages in parallel, followed by `MADV_POPULATE_READ` per page (both through ctypes, which
releases the GIL). The GPU's faults then wait on reads already in flight instead of issuing them one by one. There
is no new option and no wait on the lookup. Larger row sets keep the thread-pool touch; before Linux 5.14 (no
`MADV_POPULATE_READ`) the table falls back to the plain touch.

## Test Plan

- Unit tests (`tests/models/qwen4_exp/test_ple_pageable.py`), on the branch base (main `378504a54` wheel + #58439):
  30 passed; the two new tests (`test_fill_populates_the_rows`,
  `test_touch_falls_back_when_populate_is_unsupported`) fail on the #58439 head.
- Page fill on an idle box (57 cold pages, page cache dropped, 25 reps).
- Serving A/B on DGX Spark (GB10, sm_121, TP=1), Qwen3.8-Flash-Next, MTP n=3, KV 4 GiB,
  `checkpoint_mapped`: per fresh start, 6 different 300-token requests (cold pass), then the same 6 again (warm
  pass); 2 starts per arm, alternating; output hashes compared per request.
  Our locally converted checkpoints do not load on current main (`MergedColumnParallelLinear.load_weights` falls
  back to the module for keys it does not know), so the serving A/B ran on our serving stack (nightly
  `1ea7c63f4` + #58439) with this PR's fill path applied to the same `MappedTable.touch`.

## Test Result

Page fill, idle box (median of 25):

| fill | ms |
|---|---|
| serial touch (current) | 4.6–4.7 |
| `MADV_WILLNEED` all pages, then `MADV_POPULATE_READ` per page (this PR) | **0.36** |

Serving (ms per decode step):

| | base, start 1 / 2 | this PR, start 1 / 2 |
|---|---|---|
| cold pass (6 requests) | 59.15 / 59.41 | **56.80 / 56.50** (−2.35 / −2.91) |
| warm pass | 54.73 / 54.71 | 54.95 / 54.75 (+0.22 / +0.04) |

- Faster on 12/12 paired cold requests (−1.37 … −3.78 ms/step); major faults per step during the cold pass
  29–33 → 0.1. Output hashes identical to base in all 48 requests.
- Not measured: TP > 1; discrete GPUs (the backend targets unified memory); tables on slower storage than NVMe.

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR.
- [x] The test plan.
- [x] The test results, before and after.
- [x] Documentation update (`docs/features/engram.md`, one paragraph).

AI assistance: developed with Claude (Anthropic); the submitter reviewed every line.
