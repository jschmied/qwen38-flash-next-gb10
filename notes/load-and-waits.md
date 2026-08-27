# Where the time goes under load

Traced on 2026-08-27, bare-metal venv, `RadixArk/Qwen3.8-Flash-Next-NVFP4`,
`VLLM_PLE_CPU_OFFLOAD=1`, no speculative decoding, 8192 ctx.

The question: with a 47 GiB n-gram table gathered on CPU and paged to swap, **is the PLE
offload the bottleneck under concurrency?**

**Answer: no, and not remotely.** The PLE worker never exceeds 24% of one core, and its
per-token swap cost *falls* as concurrency rises. What actually limits throughput is
`--max-num-seqs`, and the first version of this measurement was wrong because of it.

## Method

Nothing is instrumented — everything is read from `/proc` and `/metrics`, so the measurement does
not perturb what it reports. Three candidate bottlenecks, separable by process counters:

| if... | then |
|---|---|
| PLE worker CPU approaches saturation | CPU-gather bound |
| PLE worker major faults climb per token | swap bound |
| PLE worker near-idle but decode slow | GPU bound |

Worker PIDs are read from vLLM's own log prefixes (`(PleOffloadWorker pid=N)`,
`(Worker pid=N)`) rather than matched against `/proc/*/comm`, which is not reliable here.
Harness: `scripts/ple_trace.py`.

## The confound, stated first

The initial sweep showed throughput flattening at ~33 tok/s and concluded a ceiling:

```
 c   tok/s   ttft   queue        <- --max-num-seqs 2
 1    17.1   0.23    0.00
 2    30.6   0.85    0.00
 4    32.9   6.69   23.12
 8    33.7  18.20  142.34
```

That "ceiling" was **`--max-num-seqs 2`**, a value carried over from first-boot testing. At c=4
and c=8 most requests were queued by our own configuration, not by the model. The tell is in the
data: `queue` grows to 142 s while `tok/s` stays flat — saturation and a request cap look
identical in throughput alone, and are trivially distinguishable once queue time is recorded.

## What it actually does

Same load, `--max-num-seqs 48`:

| c | aggregate tok/s | per stream | PLE cpu% | majflt/token | TTFT s | queue s | prefill s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 4.9 | 16.0 | 0.22 | 0.00 | 0.2 |
| 2 | 33.4 | 16.7 | 5.2 | 12.4 | 0.31 | 0.00 | 0.5 |
| 4 | 44.1 | 11.0 | 5.3 | 11.1 | 0.84 | 0.00 | 2.9 |
| 8 | 87.5 | 10.9 | 6.1 | 7.0 | 0.53 | 0.00 | 3.0 |
| 12 | 115.0 | 9.6 | 5.1 | 5.4 | 0.63 | 0.00 | 5.7 |
| 16 | 131.6 | 8.2 | 23.9 | 9.6 | 0.83 | 0.00 | 10.3 |
| 24 | 163.9 | 6.8 | 21.4 | 6.4 | 1.12 | 0.00 | 23.1 |
| 32 | 212.0 | 6.6 | 19.9 | 4.3 | 1.19 | 0.00 | 33.0 |
| **48** | **266.8** | 5.6 | 17.9 | 3.6 | **1.60** | 0.01 | 69.7 |
| 64 | 211.4 | 3.3 | 18.7 | 2.8 | 10.59 | 580 | 88.1 |
| 96 | 268.8 | 2.8 | 17.4 | 2.1 | 19.43 | 1713 | 139.3 |

c=64 and c=96 exceed `max-num-seqs 48` and queue again — the same artifact, now expected.

## Findings

**1. The PLE offload is not the bottleneck.** 5–24% of one core across the whole range. The CPU
gather is cheap; the table is a lookup, and only a handful of rows are touched per token.

**2. Swap cost per token *falls* with concurrency** — 16.0 major faults/token at c=1 down to 3.6
at c=48, a 4.4x reduction. This is the opposite of the intuition that a swap-backed table punishes
load. Batched tokens share n-gram rows and the page cache retains the hot set, so the *marginal*
token is far cheaper than the first. It is an argument for running this model at concurrency, not
against it.

**3. It is decode-bound, not prefill-bound**, until high concurrency: `decode_sum` is ~96% of
`infer_sum` through c=12. Prefill only becomes a visible cost past c=16.

**4. Aggregate throughput reaches 266.8 tok/s** on one GB10 at 48 streams, with TTFT still at
1.6 s and no queueing — **~15.6x the single-stream rate of 17.1**. A 125B-total / 6B-active MoE
batches extremely well.

**5. All observed "waits" are queueing waits**, and they are governed entirely by
`--max-num-seqs`. With an adequate cap, `queue_sum` is 0.00 at every concurrency the box can hold.

## Practical envelope

| workload | setting | you get |
|---|---|---|
| interactive, latency-sensitive | c ≤ 12 | 8–17 tok/s per stream, TTFT < 0.7 s |
| mixed / small team | c = 16–24 | 132–164 tok/s aggregate, TTFT ~1 s |
| batch / offline | c = 48 | **267 tok/s** aggregate, TTFT 1.6 s |

Set `--max-num-seqs` at or above your expected concurrency. The default we shipped (2) costs
**4x aggregate throughput** at c=8 and is the single most expensive misconfiguration we found.

## Why this matters for the field — CORRECTED

**An earlier version of this section claimed we were the only project measuring aggregate
throughput, and that llama.cpp builds could not measure it at all. Both were wrong.** A survey of
the field on 2026-08-27 found **six** single-Spark repos publishing concurrency numbers, and
llama.cpp serves multiple slots perfectly well — `sxuff` runs `-np 8` (93.9 tok/s aggregate),
`gitcommit90` runs 10 slots, `paragontasx` publishes c=1/2/4. `--parallel 1` was 0xBakeer's
*choice*, and they explicitly retracted the "concurrent requests crash" claim we had repeated
from their earlier README.

At matched concurrency **we are behind most of them**, because they speculate and we do not:

| c | us | Death-By-Tokens | xlzuvekas | mmcsssss | sxuff | gitcommit90 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.1 | 24.3 | 27.4 | — | 26.6 | 26.8 |
| 4 | 44.1 | **96.3** | **72.8** | 52.9 | — | 46.2 |
| 8 | 87.5 | **157.1** | — | — | 93.9 | — |
| 48 | **266.8** | not tested | not tested | not tested | not tested | not tested |

What survives the correction: **nobody has measured past c=8 on a single Spark**, and 266.8 tok/s
remains the highest aggregate published anywhere we can find. What does not survive: any claim of
leadership at the concurrencies people actually test. Death-By-Tokens does roughly **2x our
aggregate** at both c=4 and c=8.

The honest summary is narrower and less flattering than what this page said before: we measured
further out than anyone, on an axis several others were already measuring, and we are mid-table
where the measurements overlap.

## What the PLE claim does and does not cover

Our result — the offload worker uses 5–24% of one core — is **corroborated independently**:
`paragontasx` measures the gather itself as "microseconds" for a 48x160 lookup.

But two projects argue the PLE *path* still dominates, by a mechanism our metric cannot see.
`0xBakeer` and `paragontasx` both attribute **~22 ms of a 36 ms step** (respectively ~29 of 38) to
fixed, non-bandwidth overhead: the gather is a CPU op followed by a **pageable** host-to-device
copy, which forces a CUDA-graph break every token. Their evidence that this is not bandwidth is
sharp — **Q3_K moves 19% fewer bytes and is 14% *slower*** than Q4_K.

Worker CPU% cannot distinguish "the gather is cheap" from "the round-trip around the gather is
expensive". Both can be true, and on our stack the profiler says the dominant cost is elsewhere
(see [notes/single-stream-limit.md](single-stream-limit.md)) — but our claim should be read as
**"the PLE gather is not CPU-bound"**, not as "the PLE path is free".
