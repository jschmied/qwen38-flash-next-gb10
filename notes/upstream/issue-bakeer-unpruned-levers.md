Not a bug report — an arithmetic check and four levers for the **unpruned** path specifically, from
someone running the same class of box. Everything below treats your `RESULTS.md` and `LIMITATIONS.md`
as the source of truth; you flag the repo as WIP and I am not quoting it as settled.

## The cost model reproduces your measured number

| | |
| --- | --- |
| fetch / token | 0.9 GB ÷ 5.5 GB/s = **164 ms** → 6.1 tok/s storage-only ceiling |
| compute / token | 168 ms verify + 15 ms draft, ÷ acceptance 3.0 = **61 ms** → 16.4 tok/s (≈ your 15.2–15.7 resident) |
| serialised | 225 ms → **4.5 tok/s** |
| fully overlapped | 164 ms → **6.1 tok/s** |

Your unpruned measurement is 3.5–4.0. That sits just under the serialised prediction, which suggests
fetch and compute are largely **not** overlapped — and that ~35–40 % is available before any new idea.
Fetch dominates compute 2.7:1 on this path, so everything below is about bytes or bandwidth, not
kernels. Your own read that kernel tuning will not take full quality near 15 tok/s looks right.

## Four levers that keep quality exact

Excluding the three you already list (expert-major repacking, router-lookahead prefetch,
session-specific hot sets).

**1. Overlap fetch with compute, and batch.** The 4.5 → 6.1 above is the whole of it at c=1. Batching
adds a second, smaller win: at c=8 the 48 nominal slots per layer collapse to roughly 35 distinct
because routing overlaps between sequences, so bytes/token fall ~25 %. On a resident engine
concurrency *costs* expert traffic; on a streaming one the arena is shared, so it partly pays for
itself.

**2. Allocate the arena per layer by routing concentration, not globally.** A global coverage ranking
implicitly over-serves layers whose routing is flat and under-serves layers where it is concentrated.
On Qwen3.8-Flash-Next we found that concentration varies sharply by depth — **layer 0 had 7 experts
that never fire across 393,689 captured rows**, while the last four layers are a specialised tail. If
DeepSeek's router behaves similarly, a small allocation buys near-total hit rate in the degenerate
layers and those bytes are better spent where routing is flat. Computable from the trace you already
collect; no quality question.

**3. Lossless entropy coding of the FP4 stream — different from CB3.** CB3 at 3.07 bpw is lossy and
needs a quality sweep. But e2m1 has 16 codes and within a scaled group the distribution is far from
uniform: the extreme codes are rare. Empirical entropy is plausibly 3.0–3.5 bits, i.e. **12–25 % less
traffic at bit-exact quality**, and it composes with CB3 rather than competing.

**This is cheap to falsify before writing a decoder**: histogram the nibbles of two or three expert
shards and compute the empirical entropy. If it lands at 3.8 the idea is dead in minutes. If it lands
at 3.2, it justifies the per-lane PTX decoder your notes already point at — and with an exactness
guarantee the lossy path cannot offer. Same constraint either way: the decoder has to clear ~190 GB/s
to beat the FP4 kernel it replaces.

**4. More NVMe is the largest single multiplier, and it is hardware.** At 0.9 GB/token, going from
~5.5 to ~11 GB/s moves the ceiling from 6.1 to ~12 tok/s. Internal M.2 plus external over USB4 in
RAID0 is the obvious shape. Nothing in software reaches that factor on this path.

## One caution on acceptance, since it gates the pruned path too

You are using acceptance as a tuning signal. On our stack it is hypersensitive to numerics in a way
that makes cross-arm comparison unsafe:

- a kernel matching the reference to within one bf16 ulp, not bit-identical, flipped MTP acceptance by
  **+6.6, +10.4 and −3.8 pp** across three prompts (three server starts per arm, deterministic stack on);
- async scheduling on/off changed acceptance by ~1 pp while the outputs were **bit-identical — 0 of
  2,504 positions disagreeing**.

So a fast-path-vs-reference acceptance gap is expected from numerics alone and is not by itself
evidence of a quality problem. Two consequences for the decomposition you have planned: rank the arms
on tok/s rather than acceptance, and do not compare acceptance across arms that are not bit-identical.

## Context

GB10 / sm_121, one box, different model (Qwen3.8-Flash-Next), so the routing-concentration and
acceptance numbers are ours and transfer as hypotheses, not as results. Our notes are open at
[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10); your PLE-mmap
approach is what we ported for Flash-Next's n-gram table this week, so thanks for that too.

Drafted with AI assistance; the numbers were re-derived and checked by hand before posting.
