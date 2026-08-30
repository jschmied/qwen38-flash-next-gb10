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
