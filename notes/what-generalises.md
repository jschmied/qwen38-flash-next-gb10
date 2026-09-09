# What generalises

**Synthesis, not measurement.** Every number here is a pointer into a note that carries its data file;
nothing is first measured on this page. It began as an **independent reading of the repo's 50
top-level notes** (2026-09-09), commissioned by the maintainer from a model with no stake in the
conclusions — which is why it is worth keeping: it read the post-mortems and supersessions rather
than the headline numbers, and it is sharper about what this project turned into than we were. Two
of its claims were wrong against the record and are corrected below; the rest is its reading, checked.

---

## The five durable insights

1. **Agent work is TTFT- and recompute-bound. Decode tok/s is a poor optimisation target past ~35 tok/s.**
2. **QSA makes decode essentially context-length invariant. Long context is a memory and admission
   problem, not a decode-speed problem.**
3. **Byte-count rooflines are trustworthy only for large bandwidth-bound GEMMs on comparable kernels.
   Small GEMMs demand shape-level measurement.**
4. **Speculation performance is governed as much by cache and page geometry, and by numerical
   acceptance behaviour, as by drafter FLOPs.**
5. **Correctness under real batching is currently worth more than another small speed win. Sequential
   determinism is mostly solved; batch invariance is not.**

And one rule that would have prevented several dead ends in these notes:

> **An external result is evidence that a mechanism is worth testing. Never use its reported
> percentage as a prior for the expected gain.**

---

## What this repo turned into

It started as *make a 125B model fit and run fast on one GB10*. It became a study of **where LLM
serving benchmarks lie to you**. The failures that taught the most were not hardware failures — they
were claims published ahead of their evidence ([evidence-standard](evidence-standard.md): ten
withdrawn on 2026-08-31), harnesses that pinned the wrong thing
([post-mortem](post-mortem-2026-09-03.md)), and gates that voided good runs
(findings 157, 188).

| conclusion | evidence | confidence |
| --- | --- | --- |
| Agent performance is primarily a prefill / warm-recompute problem, not a decode problem | decode reached 17.1 → 36.5 tok/s, yet TTFT is 53–69 % of a turn; warm turns 2.05 → 1.52 s from one prefix-cache flag | high |
| QSA changes the long-context problem: decode is flat 4k → 60k | 26.8 → 27.1 tok/s no-spec ([depth-curve](depth-curve.md) — **see correction 1**) | medium (see below) |
| Therefore FP8 KV is a *capacity* optimisation, not a speed one | KV pool ×1.72, 4.11 → 7.07 concurrent 262k requests, decode neutral ([fp8-kv](fp8-kv.md)) | high |
| Removing bytes helps only when the op is bandwidth-bound *and* you stay on a good kernel | `lm_head` follows roofline; shared expert barely responds; hyper-connections get *slower* despite halving bytes ([lm_head](quantizing-lm-head.md), [shared expert](quantizing-shared-expert.md), [hyper-connections](why-the-hyper-connections-do-not-respond.md)) | high |
| Layer position decides whether quantisation and speculation complement or compete | FP8 dense projections reduce what MTP can amortise; FP8 `lm_head` becomes *more* valuable under MTP ([fp8-mixed-checkpoint](fp8-mixed-checkpoint.md)) | high, generalisable |
| The 47.7 GiB PLE is a memory-layout problem, not a bandwidth one | 2,560 useful bytes/token; ~26× page amplification; faults/token fall 16 → 3.6 from c=1 to c=48 ([ple-access-pattern](ple-access-pattern.md)) | high |
| Speculative-decoding acceptance is not a quality metric and is numerically fragile | a 1-ulp kernel change moved acceptance ±10 pp (finding 154); async vs sync moved it 1.1 pp with **bit-identical output** (finding 158) — **see correction 2** | high |
| Many older MTP conclusions must be discarded | `ignore_eos` added 90–100 tokens of post-EOS filler per nominal 130-token turn ([post-mortem](post-mortem-2026-09-03.md)) | high |
| Greedy correctness was a bigger problem than performance | stock: **40 distinct completions from 40 identical greedy requests** at 50k tokens; fixed: **1** (det-185) | very high |
| Concurrency nondeterminism is **not** an MTP bug | speculation fully off still perturbs 2,503/2,504 positions; sequential is exactly 0; MTP ~doubles it (det-186) | very high |
| Field *mechanisms* transfer; field *numbers* do not | bf16 SSM, MTP k=4, C-states, index sharing: all four mechanisms real, none of the reported gains reproduced (findings 153, 155, 156, 159) | high |

---

## The parts worth reading in full

**Optimise where a tensor is used, not just its size.** FP8 dense projections gave ~+39 % and FP8
`lm_head` ~+11 % without speculation, ~+19 % with MTP — good roofline candidates. But the shared
expert removed ~8 % of the byte budget for +1.9 % at c=1 while costing ~2 % NLL, and hyper-connection
`_up` removed a large byte stream and made c=1 *slower*. The profile explains it: hyper-connections
are ~25 % of GPU time because there are tens of thousands of tiny calls sitting on a ~30 µs latency
floor, not because each GEMM is expensive. Precision does not move a latency floor, and it can route
you to a worse kernel. Hence: **profile → identify shape → microbenchmark that exact shape → only
then change precision.** The repo did the opposite early on, and most false leads follow from it.

**Speculation × quantisation is not one interaction.** For dense projections they are *substitutes*:
MTP amortises the weight read across accepted tokens, and quantising leaves less to amortise. For
`lm_head` they are *complements*: the head is evaluated per draft token, so making it cheaper reduces
the marginal cost of speculation itself. Which applies depends on whether the layer runs per target
step, per verification, or per draft token.

**QSA means long context is an admission problem.** No-spec decode across 4k → 60k reads
26.8 / 26.8 / 26.8 / 26.9 / 27.1 tok/s — a fingerprint of fixed-budget sparse attention. Context then
costs memory, admission and TTFT, not steady-state decode. Spend memory savings on deeper or more
concurrent sessions; do not expect them to accelerate a session already running.

**The PLE is enormous on disk and tiny in the hot path.** 16 heads × 160 bytes = 2,560 useful bytes
per token, scattered by a modulo hash so a 160-byte fetch costs a 4 KiB page — ~26× amplification.
Batching recovers most of it (16 → 3.6 major faults/token, c=1 → c=48). So quantising the PLE buys
*resident capacity*, not decode speed: an 80-byte row still faults one page.

**The strongest current agent-speed lever is a *higher*-precision state.** `--mamba-ssm-cache-dtype
bfloat16` drops the attention block 1,600 → 832, raises KV capacity 21–36 %, and cuts paired
agent-turn TTFT **9.6 %** — not from the bytes the SSM state occupies but from hybrid allocator
geometry, the attention-vs-Mamba page coupling. It changed **127 of 2,504 modal top-1 predictions**,
so it should not ship on performance data alone; a paired coding-agent eval is the right
discriminator (finding 153).

**Determinism outranks another 10 % kernel win.** Five defects, four carried as overlays plus the
merged PLE state-stride fix, and det-184 shows they are **jointly necessary**: any one alone leaves
280–334 of ~333 disagreeing positions; all four leave zero. The practical validation is det-185's
50k-token tool-call repro — 40 identical greedy requests give 40 distinct completions on stock and 1
with the fixes. A tool *name* perturbed into invalid syntax loses an entire agent turn, which costs
more than any 5–10 % speed lever returns.

**The residual concurrent nondeterminism is a different phenomenon, and a tractable one.** Eight
concurrent requests perturb 2,503/2,504 positions with MTP entirely off, while sequential is exactly
0 — and the affected-position set **reproduces across independent server starts**. It is a
deterministic function of batch composition, not GPU noise. det-188 sharpens it: varying
`--max-num-batched-tokens` moves the flip count 119 / 234 / 132 at 2k / 4k / 16k — **peaked**, not
monotone, at the budget most incommensurate with the prompt length — and the affected sets are
**nested** (2k ⊂ 16k ⊂ 4k). One instability whose magnitude packing modulates, over a fixed set of
ill-conditioned positions. That is a far easier target than a floating one.

**"Unsupported" usually means "the guard says no".** FP8 KV was unsupported until the QSA read path
and config guard were patched; non-128 blockwise FP8 was blocked by `_WEIGHT_BLOCK_SIZE=(128,128)`
while the Triton kernel is general; mixed MoE backends were thought global and are per layer group.
**Separate hardware capability, runtime plumbing and configuration validation — a failure in the
third says almost nothing about the first.** Especially on sm_121, where much "SM120 family" dispatch
predates GB10 being characterised.

---

## Corrections applied to the original reading

1. **The QSA-flatness row cites a contaminated note, and the reading itself says so.** The 09-03
   post-mortem names `depth-curve.md` in its contamination list, and the original table nonetheless
   rated that row "high". The *no-spec* column is the part most likely to survive — no speculation
   means no acceptance to contaminate, and a decode rate over 128 tokens is robust to post-EOS filler
   if both ends are measured alike — but that argument has not been made with data. **Rated medium
   here and flagged for re-measurement.** The published flashnext page already carries a
   re-measurement caveat on parts of that curve.
2. **Findings 154 and 158 were conflated.** The original wrote that a 1-ulp kernel difference moved
   acceptance "while output quality semantics did not correspondingly change". Output was **never
   measured** in finding 154 (`pstack` ran no quality collect). Bit-identical output was measured for
   *async scheduling* in finding 158. The two are stated separately above.
3. **One claim the reading got right and we had briefly walked back.** "The exact set of affected
   positions reproduces across two independent server starts" is **correct** — Jaccard 1.000 at all
   three budgets (det-188). An earlier note of ours over-corrected this after observing that
   per-start payload hashes differ; what permutes is the assignment of outcomes to repeat indices,
   which is a different quantity.

## Where to go next

Read [README](../README.md) for the current claims and their data files,
[evidence-audit](evidence-audit.md) for what rests on how many runs (**dated 2026-09-02 — it predates
findings 141–159 and det-179–188 and needs a refresh**), [evidence-standard](evidence-standard.md)
for what counts as a finding here, and [failure-modes](failure-modes.md) plus
[post-mortem-2026-09-03](post-mortem-2026-09-03.md) for how the claims that did not survive were
caught.
