Posting this separately from the model card because two of these were surprises to us, and both are
things another person building an NVFP4 MoE calibration set would want before spending the compute.

## 1. Position matters more than project diversity

Expert routing shifts with **position inside the sequence**, not just with content. Measured on this
model: rows from positions 0–8,063 and rows from beyond 8,063 share only **56 %** of their top-50
expert set (routing total-variation 0.3747).

So a capture that stops early does not calibrate a "smaller sample of the same distribution" — it
calibrates a *different* distribution, and then those scales are applied across the model's whole
position range. Our first capture stopped at 8,063 and would have been wrong for ~64 % of the range.
The fix is to **stratify by length, not by project**: this corpus samples three length bands and
reaches position **95,239**.

If you take one thing from this post, take that. It is cheap to get right and invisible when you get
it wrong.

## 2. Record counts lie about what the Hessian actually sees

Our corpus is 221 records: 100 agent trajectories and 121 Wikipedia articles. That reads as
"mostly Wikipedia". By **tokens**, which is what the Hessian integrates over, it is the opposite:

| | items fed | tokens | share |
| --- | --- | --- | --- |
| agent trajectories (SWE-bench Multilingual) | 12 | 273,747 | **69.5 %** |
| Wikipedia, 13 writing systems | 55 | 119,942 | 30.5 % |

Length stratification is why: one agent instance (`jqlang__jq-2650`) is **95,239 tokens on its own**
and is what reaches the far end of the position range. Count tokens, not files.

## 3. If your eval and your corpus share a source, measure the overlap — do not argue it

Our held-out NLL set is Wikipedia, and our corpus is 30 % Wikipedia. The notes said the eval was
"fetched at offsets past the calibration corpus", which is a statement about how it was built, not a
measurement of what came out.

A title comparison would have been vacuous (the held-out records carry no titles). At text level,
8-gram shingles: **0 of the 21,653 held-out shingles appear among the corpus's 67,002.** Clean — but
we only know that because we checked, and the finding this repo publishes rests on that set.

## 4. The multilingual half did not fix what we added it for

We added 121 Wikipedia articles across 13 writing systems after discovering our first corpus was
**99.94 % ASCII with zero Thai**, while a build calibrated on it was corrupting Thai combining marks.
The corpus was the obvious suspect.

It was not the cause. The cause was two export-contract bugs of ours — `input_scale` written as
`amax/6` instead of `amax/2688`, and gate/up not sharing one `weight_scale_2`. Fixing those fixed the
corruption; a plain-max build with **no calibration data at all** is equally clean on the canary.

We kept the multilingual half anyway, because calibrating a 4-bit MoE only on ASCII is a bad idea
whether or not it was this bug. But we want the record straight: it is not what fixed Thai, and we
spent a day on the wrong suspect.

## 5. And calibration is not where this build's gain comes from

Stated plainly since this is the calibration thread: the held-out NLL advantage over the base
checkpoint is **`weight_scale_2` granularity** — one scale per expert instead of one per block of 128
— not the Hessian search. A plain-max per-expert build with no calibration data captures most of it.
Detail and the causal test are in the model card and in
[RadixArk discussion #13](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/discussions/13).

Corpus construction, the routing-shift measurement and the overlap check are in the open notes:
[calibration-corpus.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/calibration-corpus.md).
