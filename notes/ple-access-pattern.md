# The PLE access pattern, and why it is hard to optimise

The n-gram table dominates the checkpoint (47.7 GiB of 123) but contributes almost nothing to
per-token *bandwidth*. Understanding why explains both what it costs and which optimisations are
available.

## One lookup

| | |
|---|---|
| table | 128 shards × 2,500,012 rows × 160 dims (F8_E4M3) = **47.7 GiB** |
| row | **160 bytes** |
| rows read per token | **16** — one per n-gram head |
| useful bytes per token | **2,560** |

2.5 KB per token against a 47.7 GiB table. The table is enormous and the *read* is tiny.

## The addressing is a modulo hash — and that is the whole problem

```python
ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
```

Row index is `hash(ngram) mod head_vocab_size + head_offset`. **Uniformly distributed by
construction.** Two consequences follow, and they pull in opposite directions:

- **n-gram *frequency* is Zipfian** — a small set of n-grams accounts for most lookups.
- **n-gram *addresses* are uniform** — those hot rows are scattered evenly across 47.7 GiB.

So the hot working set is small in bytes and maximally spread in space.

## The cost is read amplification, not bandwidth

A 160-byte row costs a **4 KiB page**: **26× amplification**. Worst case per token is 16 pages =
64 KiB fetched to use 2.5 KiB.

Measured major faults per token, from our own trace:

| c | majflt/token | share of the 16 head lookups that miss |
|---:|---:|---:|
| 1 | 16.0 | ~100% |
| 8 | 7.0 | 44% |
| 16 | 9.6 | 60% |
| 32 | 4.3 | 27% |
| 48 | 3.6 | 22% |

**At c=1 essentially every head lookup faults.** By c=48 roughly 78% are served from page cache —
batched tokens share common n-grams, which is why the per-token cost *falls* with concurrency.

At ~80 µs per NVMe fault that bounds the cost at **~5% of the token budget at c=1** and ~1% at
c=16, before any overlap from the gather threads. That is consistent with our finding that the PLE
is not the bottleneck — and it also bounds the upside of any optimisation here.

## What optimisations are actually available

**Reordering rows by frequency does not work naively.** The obvious idea — pack hot rows together
so they share pages — cannot be applied by permuting the file, because the model *computes* the
address from the hash. It would need an indirection: a 20 M-entry `perm[]` (80 MB, easily resident)
consulted before the gather. That is a real change to the lookup path, and it buys at most the ~5%
above.

**Prefetch is available and unexploited.** The n-gram for the next token depends on tokens already
generated, so it cannot be prefetched during the current step — *except under speculation*, where
draft tokens are known ahead. With MTP k=2 the PLE rows for two future tokens are computable one
step early. Nobody has published this.

**Larger pages make it worse, not better.** 2 MiB huge pages would cut TLB pressure but multiply
the read amplification from 26× to 13,000×, on an access pattern that is uniform by design.

**More gather threads help, and are already the mechanism.** `VLLM_PLE_MMAP_WORKERS=32` exists
precisely so 16 independent faults overlap rather than serialise. The CPU-offload path has an
equivalent, which is why its worker sits at 5-24% CPU: it is waiting on I/O, not computing.

**Quantizing the PLE cuts the row, not the page.** NVFP4 would take the row from 160 to 80 bytes —
but a row still costs one page, so major faults are unchanged and only the *resident* footprint
improves. It buys memory, not decode. And it costs quality: measured elsewhere in the field at
worst-shard relative error 0.0345 (FP8) vs 0.1493 (NVFP4), 4.3× worse.

## The honest summary

The PLE is a small, uniformly-scattered, page-amplified read whose cost is already amortised by
batching and hidden by thread-level overlap. It is the largest object in the checkpoint and one of
the smaller terms in the decode budget. The only genuinely unexploited lever we can see is
**speculative prefetch of draft-token n-gram rows**, and its ceiling is the ~5% that faults cost at
c=1.
