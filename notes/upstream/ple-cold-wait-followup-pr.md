DRAFT — needs the user's go. vllm-project/vllm new PR, stacked on #58439 (drafted 2026-09-26 ~00:50). Code not yet in PR form: the helper must become a `csrc/` CPU op first.

Title: [Qwen4Exp][Perf] Checkpoint-mapped PLE: wait for a parallel page fill while the table is cold

## Purpose

Follow-up to #58439. With the checkpoint-mapped PLE backend, a freshly started server holds almost none of the
47.7 GiB table in the page cache: loading streams ~72 GB of weight shards through it, and 0.03 GiB of the table is
resident at ready. Each decode step then touches ~57 table pages that are not resident.

The prefetcher cannot hide this. Step N's row ids exist only after step N−1's sampling, and step N's forward starts
the gather at once, so there is zero slack. The GPU takes the faults itself, one at a time, at ~150 µs each (~60 µs
SSD read + ~90 µs fault handling): **4–5 ms per step** until enough of the table is resident, which takes minutes
of traffic.

This PR makes the lookup wait, only while pages are actually faulting, for a GIL-free parallel host page fill
(`MADV_POPULATE_READ` from 16 threads, ~95k pages/s against the SSD's ~50k IOPS parallel ceiling). The GPU then
gathers resident rows (80 µs instead of 3.7–5.5 ms). A fault-rate EMA gates the wait: it waits while ≥ 12 major
faults/step are measured, and is off once the table is warm, so a warm server keeps the host's one-step lead.

## Test Plan

DGX Spark (GB10, sm_121, TP=1), Qwen3.8-Flash-Next NVFP4, MTP n=3, KV 4 GiB (so the table does not compete with the
KV cache for memory). Nightly `1ea7c63f4` + #58439. Per start: 6 different 300-token requests on a fresh server
(cold pass), then the same 6 again (warm pass). Two start pairs, arms alternating. Output hashes compared per
request.

## Test Result

| ms/step | base, pair 1 / 2 | this PR, pair 1 / 2 | Δ |
|---|---|---|---|
| cold pass | 60.02 / 59.84 | 57.46 / 57.72 | **−2.56 / −2.12** |
| warm pass | 54.88 / 54.98 | 55.20 / 55.17 | +0.32 / +0.19 |

- Output hashes identical to base in every compared request, both pairs.
- The gate switched as designed: fault EMA 51–61/step in the cold pass (waiting), 0.0 in the warm pass (no waits).
- An unconditional wait costs +2.3 ms/step warm (host lead lost at every lookup), which is why the gate exists.
- A Python thread pool doing the same `madvise` calls was GIL-bound (3.7 ms in-server vs 2.0 ms for the C helper) and
  gave a null result.

Not measured: TP > 1; x86 discrete GPUs (the backend targets unified memory); a table on slower storage than NVMe.

Data and code (commit-pinned):
- auto-gated backend: https://github.com/jschmied/qwen38-flash-next-gb10/blob/464d9b9c7dfaeda2e062f75e5fc5b1c977ec57ae/tools/plecold/ple_pageable_synctouch_auto.py#L361-L380
- fault-rate EMA: https://github.com/jschmied/qwen38-flash-next-gb10/blob/464d9b9c7dfaeda2e062f75e5fc5b1c977ec57ae/tools/plecold/ple_pageable_synctouch_auto.py#L571
- page-fill helper: https://github.com/jschmied/qwen38-flash-next-gb10/blob/464d9b9c7dfaeda2e062f75e5fc5b1c977ec57ae/tools/plecold/fnpopulate.c
- data: https://github.com/jschmied/qwen38-flash-next-gb10/blob/5a6222e806ecf22b03d18a761b0b6e19bb8382f3/notes/data/pleauto-r12-0925.jsonl

## Essential Elements of an Effective PR Description Checklist
- [x] The purpose of the PR.
- [x] The test plan.
- [x] The test results, before and after.
- [ ] (Optional) Documentation update — the #58439 docs section gets one paragraph.

AI assistance: developed with Claude; every line reviewed by the human submitter.

---
Before opening (not part of the body):
- Port `fnpopulate.c` into a `csrc/` CPU op (or `torch.ops` via the existing cpu extension), no ctypes.
- Replace the env var with an `--engram-config` key (`"cold_wait": "auto" | "on" | "off"`), default `auto`.
- Unit test: gate off below threshold, on above; no wait when every page is resident.
- Check the line anchors against the pinned file before posting; rebase on #58439's head.
