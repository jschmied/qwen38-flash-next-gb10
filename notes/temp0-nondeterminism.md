# Temperature 0 is not reproducible on this model, from ~30 tokens on

Measured 2026-08-30. One GB10, FP8-head checkpoint, `--max-model-len 65536`, **c=1 and strictly
sequential** (one request in flight), **speculation off and verified absent from the server log**,
identical ~5,700-token prompt, `temperature: 0`, thinking disabled so content starts immediately.

| max_tokens | distinct outputs of 5 | chars | first divergence |
|---:|---:|---:|---:|
| 32 | **3** | 161–167 | char 52 |
| 128 | **5** | 592–647 | char 29 |
| 512 | **5** | 1,861–2,143 | char 9 |
| 2,000 | **5** | 6,744–7,441 | char 9 |

## What this rules out

Three suspects were eliminated before this table, each with its own control:

- **The prefix cache.** In the 8-request probe, requests 1 and 2 had *zero* cache hits and still
  differed from each other. Also 7/8 vs 6/8 empties across a caching-on/off pair — no separation.
- **Speculation.** The same 8/8-distinct result with MTP off, with the server log checked for a
  speculative config before the run rather than after.
- **Concurrency.** Requests are issued sequentially from a Python loop; nothing else is on the box.

## What it says

**Divergence is not a compounding effect.** At 512 and 2,000 tokens it begins at character 9 — the
outputs separate almost at once. At 32 tokens only 3 of 5 differ, so it is probabilistic per token
rather than certain, but by 128 tokens all five differ. There is no reproducible regime above
roughly thirty tokens.

## This contradicts our own 27B result, and that is the interesting part

`[[temp0-not-reproducible-under-load]]` records the Qwen3.8-**27B** as byte-identical at temperature
0 across three runs of 1,719 characters, diverging *only* when batched with concurrent filler. Its
headline — "temperature 0 is reproducible only at batch size 1" — is true there and **false here**,
at batch size 1.

**The candidate mechanism is MoE routing, and it is the obvious structural difference.** The 27B is
dense; this model is a 125B-total / 6B-active MoE. A dense model's non-determinism has to come from
reduction order inside a GEMM, which only flips an output when an argmax is already a near-tie. An
MoE adds a much larger amplifier *upstream* of that: top-k expert selection. Two experts with
near-equal gate logits can swap on the last bits, and then a different pair of experts computes the
token — not a slightly different sum, a different function. That is a far bigger perturbation than
reduction order, which fits divergence appearing at character 9 rather than after hundreds of
tokens.

This is a hypothesis with a clean structural argument, not a measured cause. Testing it would mean
logging the routing decisions for the same token position across runs and checking whether the
selected expert set differs — worth doing, and not done yet.

## The FP8 `lm_head` is not the cause — controlled and refuted

Logprob *differences* at a position land on an exact 1/16 grid (`-33/8`, `-27/4`, `-113/16`), while
the raw values are full precision. That looked like an FP8-head signature, and it suggested a clean
story: coarse logits → exact top-2 ties → order-dependent argmax.

**Both halves of that were wrong, and the checkpoint pair proves it.** `fp8mix` and `fp8head` differ
in exactly one thing — BF16 versus `FP8_PB_WO` head — with `FP8_PB_WO` dense projections in both:

| | fp8head (FP8 head) | fp8mix (BF16 head) |
|---|---|---|
| pos-0 logprob differences | −33/8, −27/4, −113/16, −117/16 | −29/8, −27/4, −115/16, −29/4 |
| distinct of 5 at 32 tok | 3 | **5** |
| distinct of 5 at 128 / 512 / 2000 | 5 / 5 / 5 | **5 / 5 / 5** |

1. **The grid is BF16 resolution, not FP8.** A BF16 significand gives ULP `2^-4 = 1/16` for values
   in `[8,16)`, which is where these logits sit. The grid appears identically with an unquantized
   head, and `fp8mix` even shows an exact **three-way tie** (`' '`, `'.'`, `'3'` all at
   `-10.187644004821777`). Ties are a normal consequence of BF16 logits.
2. **Determinism does not return with a BF16 head.** Five distinct outputs of five at every length.

So the FP8 `lm_head` is exonerated: it neither creates the grid nor the divergence. Its cost remains
what we measured — **+19% under speculation, no measurable quality cost** — and reproducibility is
not a hidden price it charges.

**What survives:** divergence is present with an FP8 head *and* a BF16 head, with speculation on
*and* off, with cache hits *and* none, at c=1 sequential. It is intrinsic to this model on this
stack. MoE routing remains the leading structural candidate purely because it is the difference
between this model and the dense 27B that *is* reproducible — but that is an argument from structure,
not a measurement, and the head hypothesis just showed how far a plausible structural argument can
be from the truth.

## Operational consequences

- **No text-identity test on this model.** Regression suites, eval harnesses and any A/B that
  compares runs by output equality are invalid here regardless of temperature, concurrency or
  speculation. Score with aggregate metrics over many samples, or teacher-forced logprobs, which
  force the token sequence and so cannot be flipped by an argmax.
- **`temperature: 0` is not a determinism switch.** It is worth saying plainly because the whole
  field writes it as though it were.
- **A field claim to treat carefully:** another recipe reports three identical temperature-0
  requests returning byte-identical answers on vLLM/GB10. That is not reproducible here at any
  length we tested. Their checkpoint differs from ours (no FP8 dense projections, no FP8
  `lm_head`), so the comparison is not controlled — but it does mean "temp 0 is stable on this
  model" is a **checkpoint-scoped** claim, not a model-scoped one.

## The `indexer_budget` hypothesis — someone else's, and it does not hold here (2026-08-31)

vllm#54521 reports the same symptom on the same build and proposes a mechanism we had not
considered: greedy decoding deterministic **below** `indexer_budget` and non-deterministic above,
because QSA switches from dense attention to top-k selection at that point. Every probe of ours had
used ~5,700-token prompts — above the budget — so it would have explained our results neatly.

**It does not survive a controlled test here.** Five identical requests per cell, `ignore_eos`
forcing generation length, `indexer_budget = 2048`:

| prompt_tokens | vs budget | MTP k=2 | MTP off |
|---:|---|---|---|
| 582 | **below** | 3 of 5 | **3 of 5** |
| 1,142 | **below** | 3 of 5 | **4 of 5** |
| 1,982 | near | 5 of 5 | 4 of 5 |
| 3,102 | above | 4 of 5 | 2 of 5 |
| 5,622 | above | 3 of 5 | 3 of 5 |

Divergence at 582 prompt tokens, under a third of the budget, and it survives disabling speculation
— which was the obvious confound since we normally run MTP k=2.

**So the count of eliminated hypotheses is now four:** prefix cache, speculation, the FP8 `lm_head`
(single-variable control), and the QSA top-k threshold. Reported on #54521 with the two differences
that could still explain the disagreement — our derived checkpoint versus their stock one, and our
patched tree — plus an offer to run any cell they want on this hardware.

## A fifth candidate, and this one has code behind it (2026-08-31, later)

Not ours — it came out of a sweep of upstream issues, and it is the first candidate that explains
*every* observation rather than being merely consistent with some of them.

**On sm_121 we never run the cooperative top-k.** `nvidia/ops/qsa.py:804-813` selects the kernel:

```python
use_cooperative_topk = (
    current_platform.has_device_capability(90)
    and not current_platform.is_device_capability_family(120)
)
... use_cooperative_topk ... else torch.ops._C.persistent_topk
```

`is_device_capability_family(120)` is `(cap // 10) == (capability // 10)`, so it matches **all of
12.x**, GB10 included. Verified on the box:

    capability                      : DeviceCapability(major=12, minor=1)
    has_device_capability(90)       : True
    is_device_capability_family(120): True
    => use_cooperative_topk         : False

So every request here takes `torch.ops._C.persistent_topk` — and that is the op of **vllm#51782**,
which reports it **silently returns wrong values** when many row elements share one coarse
histogram bin. The report is specific in ways that matter: the error is **data-dependent, not
shape-dependent**; there are no `-1` slots and no duplicate indices, so it is invisible to every
structural check; and a third dispatch path (`histogram_2048_topk`, tie buffer `DBUF = 3708`) is
active for `seq_len <= 8192` — **below** `indexer_budget`.

That last detail is the one that matters, because it is exactly the regime where we measured
divergence and #54521's model predicted none. It also fits what the eliminations left standing: a
data-dependent kernel fault is invisible to a prefix-cache control, survives MTP-off, is unaffected
by the `lm_head` dtype, and has nothing to do with the top-k *threshold* — only with the top-k
*kernel*. @Leonccaa reports the same on #51782 for this model at ≥4K context, diverging from
token 2.

**Status: candidate, not conclusion.** We have not demonstrated the wrong values ourselves. The
discriminating test is to compare `persistent_topk`'s output against a reference top-k on the same
inputs — a correctness test, not a determinism test, which is a different harness from anything
here. Until that runs, this is a well-supported hypothesis with a verified selection path, and the
mechanism is still formally open.

**What it would mean if it holds:** #54521 is a duplicate of #51782 scoped to sm_121 by the
family-120 exclusion, and the sub-budget divergence we reported is the evidence separating them.

## The family-gate pattern is the real lesson

This is the third capability-*family* check to bite this box, and they all fail the same way —
`major`-level or family-level tests that treat sm_121 as something it is not:

| gate | where | consequence |
|---|---|---|
| `is_device_capability_family(120)` excludes 12.x from cooperative top-k | `nvidia/ops/qsa.py:804` | forced onto the buggy `persistent_topk` — above |
| `is_arch_support_pdl()` is `major >= 9` | `platforms/cuda.py:712` | PDL used in `_build_qsa_metadata_kernel`; the dependent kernel waits forever → the >8k hangs of vllm#53960 |
| `support_deep_gemm()` accepts the whole 120 family | — | DeepGEMM reported supported on sm_121, then faults (vllm#54125). We already set `VLLM_USE_DEEP_GEMM=0` |

Two of the three are conservative-but-wrong and one is permissive-but-wrong, so there is no safe
direction to guess in. **A capability family is not a capability**, and on this hardware that
distinction has produced a silent correctness bug, a deadlock, and a fault.

**Mechanism: still open, but no longer without a suspect.** Four explanations were measured and
discarded; the fifth arrived with a verified dispatch path and a matching upstream report, and now
needs a correctness harness rather than another determinism run.
