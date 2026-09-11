# The Local-Hessian rebuild corrupts Thai combining marks — 2026-09-10

**The build is verified correct and measurably worse.** Every structural check passed; the quality
check failed. That is the whole point of having both.

## Result

| arm | starts | responses | corrupt | exact copies |
| --- | --- | --- | --- | --- |
| **stock** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) | 2 | 72 | **0** | **72 / 72** |
| **LH rebuild** (`/opt/llm/models/fnext-lh`) | 1 | 24 | **6** | 12 / 24 |

Identical server flags both arms (`FN_MAXLEN=8192 FN_SEQS=4 FN_UTIL=0.75 FN_MTP=3 FN_SPEC_NODROP=1`,
port 8092, `flashnext`), same prompts, temp 0, seed = rep index. The **only** differing cell is
`FN_MODEL`.

## The damage is structural, not a wording difference

```
stock:  ผู้ใช้ต้องเข้าสู่ระบบก่อนที่จะแก้ไขข้อมูลนี้ได้
ours:   ผู้ใช้ต้องเข้าสู่ระบบก่่นที่จะแก้ไขข้อมูลน้ีได้
```

Two distinct faults in one line: `ก่อน` loses its base character **อ** and gains a **doubled mai ek**
(`่่`), and `นี้` comes back as `น้ี` with vowel and tone mark **reordered**. That is exactly the
ordering damage MiaAI reported in their #27.

## Why this is not the model's known nondeterminism

`temp0-not-reproducible-under-load` records that Flash-Next diverges run to run even at c=1. Three
things rule that out here:

1. **Deterministic per prompt within the arm** — th-1 3/3 corrupt, th-3 3/3 corrupt, th-2 0/3.
2. **Stock is flawless across two starts** — 72/72 exact. Divergence would not spare it entirely.
3. **The exact-copy rate halves**, 72/72 → 12/24. That is a broad capability difference, not a
   coin-flip on one token.

## Scope

Thai only. Devanagari, Arabic harakat, Hebrew niqqud and emoji ZWJ were clean in both arms. The
Mn/Mc/Me census correctly declined to score the Devanagari rewordings as corruption — the
discrimination that was tested synthetically before the probe was trusted.

## What it means for MiaAI #27 / our #37

**It cuts against our own hypothesis.** Our issue #37 proposed that #27's corruption comes from the
shipped build quantizing the GDN/linear-attention path with a five-month-old ModelOpt. Here the
signature appears in a build whose `linear_attn` is **excluded from quantization** — the damage came
from quantizing **experts**. So expert quantization is *sufficient* to produce this signature, and
#37's mechanism is not necessary for it. That does not disprove #37, but it removes this instance as
support and weakens the case. Worth telling them; needs the user's go.

## Consequences taken

- **Upload killed mid-transfer.** `upload_folder` commits once at the end, so the commit never
  landed: **no weights were ever published.** The repo has only ever held the card and the verifier.
- The public card now carries a ⛔ banner with these numbers and the example above.
- SWE run A not started — the canary failed first, and at n=10 it could not have detected this
  anyway (SE ~15 points against a defect the canary caught in 35 minutes).

## Open: the cause

Not identified. Candidates, cheapest first:

1. **The 33 no-rows experts** on plain-max — but that is the standard method every published build
   uses, so it should not be worse than stock.
2. **The 58 thin experts** (1–63 routed rows) — under-determined Hessians could be actively worse
   than plain max, which would make LH harmful exactly where data is scarce.
3. **A packing defect** — E2M1 encoding, low-nibble-first order, or the `input_scale` convention
   (we set `input_amax/6`; if vLLM expects `amax/448` the activations are off by 74.7×, though the
   output is far too coherent for that).
4. **The gate/up split** — we assume `gate_up_proj` is `[gate; up]` along dim 1. Reversed or
   interleaved would corrupt SwiGLU systematically.

**Discriminating test:** rebuild one layer with plain max instead of Local-Hessian and re-probe. If
plain max is clean, the calibration is at fault; if it also corrupts, the packer is.

## ~~CAUSE FOUND — calibration-set overfitting~~ **REFUTED 2026-09-10 20:52 — see the verdict below**

### The packer is exonerated, independently

Reconstruction against the BF16 truth, layer 24, 18 (expert, matrix) pairs:

| | mean error vs BF16 |
| --- | --- |
| **ours (Local-Hessian)** | **8.585 %** |
| RadixArk | 9.498 % |
| **our plain-max** | **9.494 %** |

**Our plain-max matches RadixArk's to 0.004 pp.** That validates the E2M1 encoding, the low-nibble
order, the group size, and the scale conventions against a third-party implementation — the packer
writes correct bytes. And Local-Hessian does what it claims: **0.91 pp better** than both.

So the rebuilt weights are **measurably closer to BF16 and behaviourally worse.** That eliminates the
packer, the gate/up split and the `input_scale` convention in one measurement.

### The corpus has no Thai in it. None.

```
calibration corpus: 5,775,370 characters
  ascii            5,772,067   99.9428 %
  other-nonascii       3,303    0.0572 %      <- and zero Thai
```

Local-Hessian minimises `dw·H·dwᵀ` with **H estimated from our activations**. Directions that carry
Thai have ~zero energy in that H, so the search spends their precision on the directions that
dominate English and code. Global L2 improves *because* of the trade. Thai pays for it.

This is textbook calibration-set overfitting, with an unusually clean demonstration: better global
reconstruction, catastrophic on a script the corpus never contains.

### The mistake underneath it was mine, and it is a naming trap

I described this corpus as spanning "roughly ten languages — Java, JS/TS, Go, Rust, Ruby, PHP,
C/C++, Python, Lua, jq" — on the model card, in `calibration-corpus.md`, and repeatedly in
conversation. **Those are programming languages.** SWE-bench *Multilingual* is multilingual in
**code** and monolingual in **human script**: 99.94 % ASCII. I never checked, because the dataset's
name had already answered the question for me.

Every stratification decision compounded it. I measured position coverage, project diversity, expert
row counts — and never once counted a codepoint outside ASCII.

### The fix is the corpus, not the method

Local-Hessian is working. It needs a calibration set that contains what we do not want it to
sacrifice. Concretely: add natural-language text across scripts — Thai, Devanagari, Arabic, Hebrew,
CJK, Cyrillic — to the agent trajectories, then rebuild.

**Testable prediction:** a **plain-max** build should be clean on Thai, because plain max is
data-independent. If it corrupts too, this diagnosis is wrong. That test is queued and is cheap: the
merge is index-only, so one swapped layer file is enough to start.

**Second prediction:** the corrupted scripts should be exactly those absent from the corpus, and the
clean ones (Devanagari, Arabic, Hebrew, ZWJ were clean) should show damage too once probed harder —
they are equally absent. Their cleanliness may just mean the probe's prompts were easier.

## Is MiaAI #27 the same mechanism? — probably, and for a structural reason

Three builds, same architecture, one coherent story:

| build | `quant_algo` | ModelOpt | data-dependent calibration? | Thai |
| --- | --- | --- | --- | --- |
| `RadixArk/…-NVFP4` | `NVFP4`, uniform, group 16 | 0.46.0 | need not be — and measurement says it is not | **clean 0/72** |
| **ours v1** | NVFP4 + Local-Hessian | 0.46.0 | **yes**, on a 99.94 % ASCII corpus | **18/72 corrupt** |
| `local-inference-lab/…-NVFP4` | **`MIXED_PRECISION`** | 0.39.0.dev290 | **yes, by construction** | corrupt (#27) |

**Why RadixArk is inferred data-independent:** its reconstruction error against BF16 is **9.498 %**
and our own plain-max is **9.494 %** — 0.004 pp apart. A data-dependent method would not land that
close to plain max by accident. Plain max cannot overfit a corpus because it never looks at one.

**Why local-inference-lab is data-dependent by construction:** `MIXED_PRECISION` *requires*
calibration data — it chooses which layers get MXFP8 vs NVFP4 vs W4A16 (467 / 48 / 29 in their case)
by measuring sensitivity on a calibration set. Uniform NVFP4 at a fixed group size does not.

So the axis that separates clean from corrupt across all three builds is **whether the calibration
was data-dependent**, not which module was quantized.

### What this does to our own issue #37

**It weakens it further, and we should say so.** #37 proposed that #27's corruption comes from the
shipped build quantizing the GDN/linear-attention path with a five-month-old ModelOpt. That
hypothesis never had a demonstrated mechanism. Calibration-set overfitting now does — a controlled
run on the same architecture, changing exactly one variable, reproducing the same codepoint
signature — and it explains a case #37 cannot: **our build does not quantize `linear_attn` at all**
and corrupts anyway.

#37 is not disproven; old ModelOpt and GDN quantization could still contribute. But it is no longer
the best explanation, and the honest thing is to tell them that rather than let our issue stand at
its original confidence.

### Limits, stated

- We cannot see local-inference-lab's calibration corpus. Their method's data-*dependence* is
  established from `MIXED_PRECISION`; the corpus *content* is inferred.
- RadixArk's plain-max attribution rests on a 0.004 pp reconstruction match, not on a statement from
  them.
- Both are testable by the v2 rebuild now running: if adding script coverage fixes ours, the
  mechanism is confirmed on at least one build.

---

# VERDICT: the overfitting diagnosis is REFUTED

I predicted that adding script coverage to the corpus would fix this. It made it **twice as bad**.

| arm | corpus | corrupt | exact | scripts hit |
| --- | --- | --- | --- | --- |
| stock (RadixArk), 2 starts | — | **0/72 (0 %)** | **72/72** | — |
| v1, 2 starts | 99.94 % ASCII, zero Thai | 18/72 (25 %) | 36/72 | Thai |
| **v2, 1 start** | **30 % script, 13 languages** | **24/48 (50 %)** | 18/48 | **Thai + Devanagari** |

**The decisive detail is not the rate, it is which scripts broke.** Devanagari was clean in both v1
arms. I added Devanagari to the corpus, and Devanagari started corrupting — `U+093E` ×2 and `U+0902`
on hi-1, `U+0902` ×3 on hi-2, deterministically on all six reps.

No version of "the corpus lacked coverage" predicts that giving a script coverage breaks it. The
hypothesis is dead, not wounded.

I also got the secondary prediction backwards. I wrote that Devanagari/Arabic/Hebrew "should show
damage too once probed harder — they are equally absent." They were not absent in v2; they were
present, and that is when Devanagari failed.

## What the evidence now supports

The build quality *improved* by every internal measure while behaviour got worse:

| | v1 | v2 |
| --- | --- | --- |
| full-Hessian experts | 99.63 % | **99.83 %** |
| thin (1–63 rows) | 58 | **32** |
| no-rows | 33 | **11** |
| reconstruction vs BF16 (layer 24) | 8.585 % (v1 measured) | — |
| **corrupt responses** | **25 %** | **50 %** |

More rows per expert, fewer under-determined Hessians, better global reconstruction — and twice the
corruption. That pattern points away from *what data* Local-Hessian sees and toward **Local-Hessian
itself**, or the export path around it.

## Next test — and it is cheap and needs no capture at all

**Plain-max build.** Plain max is data-independent: it takes `amax` over the weights and never looks
at an activation. So it needs **no capture and no forward pass** — just pack and export.

- **plain-max clean** -> Local-Hessian is at fault, not its data. Our plain-max reconstruction already
  matches RadixArk's to 0.004 pp, so a plain-max build should behave like RadixArk; if it does, the
  packer and export path are exonerated end to end and LH is the sole suspect.
- **plain-max also corrupts** -> the fault is in our packer or export path, *despite* matching
  RadixArk's reconstruction to 0.004 pp. That would mean reconstruction error is blind to whatever is
  breaking, which is itself the finding.

Either outcome is decisive, which is what the previous experiment was not.

## Standing correction to the cross-build story

The note above argued MiaAI #27 is probably the same mechanism, keyed on data-dependent calibration.
**That argument rests on a diagnosis that has now been refuted** and should not be carried to them.
The observation that survives is narrower and still worth something: RadixArk (reconstruction 9.498 %,
i.e. plain-max-like) is clean, and two data-dependent builds are not. But "data-dependent" is no
longer a demonstrated mechanism — v2 was *more* data-dependent and *more* broken.

---

# ROOT CAUSE — two producer/runtime contract violations in our export

**Found by the user, 2026-09-10 21:0x, and confirmed against RadixArk within minutes.** Not
Local-Hessian, not the corpus. Both of my diagnoses were wrong; this one is measured on both sides.

## Contract 1 — gate and up must share one `weight_scale_2`

vLLM fuses gate and up into `w13`, and on a mismatch **silently applies the GATE scale to both
halves** ([vllm#54974](https://github.com/vllm-project/vllm/issues/54974)). Our builder calibrated
them separately, so each got its own:

| | gate/up `weight_scale_2` mismatches, layer 24 |
| --- | --- |
| RadixArk | **0 / 512** |
| ours | **506 / 512** |

Ratio spread across those experts: min 0.359, median **1.216**, p90 1.754, max **3.185**;
**383/512 more than 10 % off**. So the up projection of three quarters of every layer's experts was
being dequantized at runtime with a global scale up to 3× wrong.

## Contract 2 — `input_scale` is `amax / (E2M1_MAX * E4M3_MAX)` = `amax / 2688`

Not `amax/6`. ModelOpt states the convention explicitly
([config.py](https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/config.py)),
and TRT-LLM describes it as `amax/(448*6)`.

| build | what we wrote | vs RadixArk's 0.00246466 |
| --- | --- | --- |
| v1 / v2 (LH) | `amax / 6` | **448× too large** |
| plain-max | `1.0` | **406× too large** |

Back-solving RadixArk's value confirms the convention: `0.00246466 × 2688 = 6.63`, a plausible
hidden-state amax; under `/6` it would imply 0.0148, which is not.

## Why every check we had missed both

**Our reconstruction test measured weights the runtime never uses.** It decoded each matrix with
*its own* `weight_scale_2` — so gate reconstructed perfectly, up reconstructed perfectly, and the
runtime then threw up's scale away. The 0.004 pp agreement with RadixArk on plain-max was real and
irrelevant: it says nothing about `input_scale`, and nothing about a scale the runtime discards.

That is why the symptoms looked paradoxical and I kept blaming the calibration:

| symptom | explained by |
| --- | --- |
| weight reconstruction excellent | test used each matrix's own scale |
| output mostly coherent | `dequant_alpha = weight_scale_2 * input_scale` cancels much of the global error |
| small deterministic errors on rare tokens | activation FP4 range/clipping changed |
| **v2 worse than v1** | different activations → different amax → different wrongness, *not* "LH learned Devanagari badly" |

## Status

Both fixed and verified on a rebuilt layer 24: **0/512** gate/up mismatches, **0/1536** input_scale
values differing from the base. `input_scale` is now **copied from the base checkpoint** rather than
recomputed — a weight-only rebuild does not change the activation distribution, so copying removes
the whole class of error rather than swapping one constant for another.

Full 48-layer rebuild running. Local-Hessian is **not** exonerated yet — it is merely no longer the
prime suspect, and cannot be judged until a build that satisfies both contracts is measured.

## Full-model scale of the defect, and a verifier level that catches it

`mergeverify.py` now has a **LEVEL 1.5 — producer/runtime contracts**, run against the known-bad v2
merge as a negative control:

```
FAIL  24,357/24,576 gate/up weight_scale_2 MISMATCH   (99.1 % of all experts)
FAIL  73,728/73,728 input_scale values differ from the base   (100 %)
```

Every `input_scale` in the model was wrong, and the up projection of 99.1 % of all experts was being
dequantized with a scale the runtime discards.

**Why this level had to exist.** Levels 1 and 2 both PASSED on that same checkpoint. They verify
names, completeness and per-tensor provenance — and they reconstruct each matrix with *its own*
scale. That is not what the runtime does. A verifier that never models the consumer's contract can
be perfectly green on a checkpoint that is 99 % wrong where it matters.

The general form, worth carrying beyond this project: **a producer-side check validates the file; it
does not validate the file's agreement with the consumer.** Test against the consumer's contract, or
against a known-good third-party artifact — RadixArk's 0/24,576 was what made both defects visible in
minutes once we thought to look.

---

# RESOLVED — v3 is clean. The export contracts were the whole story.

| arm | corpus | contracts | corrupt | exact | scripts hit |
| --- | --- | --- | --- | --- | --- |
| stock (RadixArk), 2 starts | — | correct | **0/72** | **72/72** | — |
| v1, 2 starts | 99.94 % ASCII | broken | 18/72 (25 %) | 36/72 | Thai |
| v2 | 30 % script, 13 langs | broken | 24/48 (50 %) | 18/48 | Thai + Devanagari |
| **v3** | **same as v2** | **fixed** | **0/48** | **48/48** | **—** |

**v2 → v3 changed nothing but the two export contracts** — same capture, same Local-Hessian,
identical calibration buckets (thin 32, no-rows 11, 24,533/24,576 full-Hessian). 50 % → 0 %.

And it is genuinely the rebuilt model, not a silent fallback: level 3 gives 94/96, 91/91, 85/85 and
50/50 tokens diverging from stock, max |Δlogprob| 1.62, coherent output and a correct tool call in
every cell.

## Scoreboard, stated plainly

Three diagnoses of mine, all wrong:

1. **"The packer is exonerated"** — from a reconstruction test that decoded each matrix with its own
   scale, i.e. weights the runtime never uses. The 0.004 pp agreement with RadixArk was real and
   irrelevant.
2. **"Calibration-set overfitting; add script coverage"** — cost a corpus rebuild and 90 minutes of
   GPU. Refuted by its own result: corruption doubled and spread to the script I had just added.
3. **"Local-Hessian itself is at fault"** — escalating to the most sophisticated component after the
   second failure, instead of reopening the plumbing I had declared clean.

The user's diagnosis, right on both counts and confirmed against RadixArk in 90 seconds:
`input_scale` must be `amax/(6*448)`, and gate/up must share one `weight_scale_2`.

**Local-Hessian was never at fault. The corpus was never at fault.** Both v1 and v2 were mis-exported,
so neither measured what I claimed. The corpus v2 work was not wasted — script coverage is defensible
on its own terms — but it was undertaken for a reason the evidence had already contradicted.

## What made the difference, methodologically

Every check I built compared our output to **our own intent**. The user's compared it to **the
consumer's contract** and to a **known-good third-party artifact**. `mergeverify.py` LEVEL 1.5 now
does the same and is exercised against the bad build so it is proven to fire.

Recorded as a durable lesson in memory `diff-against-known-good-artifact`.

---

# Is Local-Hessian worth anything? — not demonstrably, 2026-09-11 01:30

With the export contracts fixed, a **plain-max** build (data-independent, no capture, no Hessian) was
built, merged and probed under identical conditions.

| arm | calibration | corrupt | exact |
| --- | --- | --- | --- |
| stock (RadixArk), 2 starts | plain-max-like | 0/72 | 72/72 |
| **v3** | **Local-Hessian** | **0/48** | **48/48** |
| **pmax** | **plain max** | **0/48** | **48/48** |

**Both clean. The contract bugs were the entire story, and Local-Hessian is not required for
correctness.**

Not a null from identical builds — pmax vs v3 diverge on **93/96, 89/89, 80/80 and 49/49** tokens,
max |Δlogprob| **1.73**. Two substantially different sets of weights that behave equally well on the
only quality measure we have.

## What Local-Hessian has actually demonstrated

**One number: 8.585 % vs 9.494 % weight reconstruction against BF16** (layer 24, 18 expert/matrix
pairs) — 0.91 pp, in weight space.

**No behavioural benefit has been shown.** The canary cannot distinguish them; nothing else has been
measured. Weight reconstruction has already proven a poor predictor here — it was 0.004 pp between
our plain-max and RadixArk while two contracts were broken, and it reported v1's Local-Hessian build
as *better* than stock while that build corrupted 25 % of Thai responses.

## Consequence for the model card

The card led on Local-Hessian as the reason the repo exists. That claim is not supported by anything
measured, so it has been **softened rather than left standing** on a public page. What can honestly
be claimed: the build is correct, verified, and equal to stock on the canary; LH gives 0.9 pp better
weight reconstruction; whether that is worth anything is unmeasured.

**What would earn the claim back:** NLL divergence against BF16 on held-out text, LH vs plain-max vs
stock. That is the measurement `kv-dtype-logprob-experiment` was designed for and it is the only
thing that would justify shipping LH over a build anyone can make with no calibration data at all.

---

# Held-out NLL, three arms — 2026-09-11 02:55

15 Wikipedia passages / 15,880 tokens, fetched at offsets far past the calibration corpus, none of it
seen by any arm. Fixed text, not a generated continuation. Threshold set **before** looking: a gap
under ~0.005 on this sample is not a result.

| group | pmax | lh_v3 | stock | best |
| --- | --- | --- | --- | --- |
| Devanagari | 2.7765 | **2.7646** | 2.8557 | lh_v3 |
| Thai | 2.2435 | **2.2233** | 2.2337 | lh_v3 |
| English | 2.0037 | **1.9947** | 2.0076 | lh_v3 |
| Arabic | **2.2776** | 2.2780 | 2.2884 | pmax |
| Cyrillic | **1.3890** | 1.3989 | 1.4022 | pmax |
| Latin | **1.6576** | 1.6578 | 1.6591 | pmax |
| Hebrew | 1.3070 | 1.3014 | **1.2970** | stock |
| **overall** | 1.9542 | **1.9488** | 1.9663 | |

## Two conclusions, one firm and one not

**FIRM — both rebuilds beat the base checkpoint.** `lh_v3` −0.0175 and `pmax` −0.0121 against stock,
both clearing the threshold, and robust rather than outlier-driven: **12/15 passages** favour one of
ours, medians agree with means (1.9158 / 1.9352 / 1.9422), and the largest gap (Devanagari) is
consistent across *both* its passages (−0.1036, −0.0900).

**NOT ESTABLISHED — that Local-Hessian is the reason.** `lh_v3` beats `pmax` by **0.0054**, which is
exactly at the threshold. Directionally favourable, three groups each, LH's winning margins larger
than pmax's. Call it suggestive; do not call it a result.

## The finding that actually matters here

**`pmax` has no calibration data at all and still beats stock by 0.0121.** So the improvement over
RadixArk does not come from calibration — not from Local-Hessian, and not from the corpus either.

**Cause unknown.** Our plain-max reconstruction matched RadixArk's to 0.004 pp, so it is not a gross
weight difference. Candidates, none tested:

- `weight_scale_2` granularity or derivation. We take the max amax across a gate/up pair and divide by
  2688; RadixArk shares a scale too, but may derive it differently.
- Group-level amax details — clipping, tie handling, the E4M3 rounding of `weight_scale`.
- `down_proj` scale treatment, which is per-matrix in both but need not agree.

The honest position is that our export differs from RadixArk's in some way that helps slightly on
held-out text, concentrated on the hardest script in the set, and we have not identified it. That is
worth chasing precisely **because** it is not the mechanism we set out to test — and after today, a
difference we cannot name is a difference we should not ship claims about.

## Where our export differs from RadixArk's — named, but not shown to be the cause

Offline comparison of layer 24, our plain-max scale structure against RadixArk's own export
(`scalediff.py`, no GPU):

| matrix | scale_2 ratio ours/theirs | `weight_scale` differing | packed nibbles differing |
| --- | --- | --- | --- |
| `down_proj` | **1.0000** | **0.00 %** | 0.1–0.45 % |
| `gate_proj` / `up_proj` | 0.17–0.40 | **100 %** | ~12.5 % |

**Our unfused export is byte-exact with theirs.** `down_proj` gets the identical `scale_2` and
byte-identical `weight_scale`, which independently confirms our amax derivation, the `/2688`
convention and the E4M3 rounding. The entire difference is in the **fused gate/up pair**.

**And the difference is granularity:** RadixArk uses **3 distinct gate/up `scale_2` values across all
512 experts** — in blocks of 256/128/128, so presumably batched over shards — while we derive one per
expert. Our per-expert scales fit each expert tighter.

**Two hypotheses tested and refuted along the way:**

- *E4M3 subnormals* — a larger `scale_2` pushes `weight_scale` down, so maybe theirs lose relative
  precision. **No:** 0 % subnormal in both, medians 88–128 (ours) vs 28 (theirs), all inside the
  normal range where E4M3's relative precision is uniform.
- *Better weight reconstruction* — **too small to matter:** ours is 9.4912 % vs 9.4958 %, a
  **0.0046 pp** edge against a held-out NLL gap of **0.0121**, two and a half times larger.

**So the cause of the NLL improvement over RadixArk remains unidentified.** We have a real structural
difference and a real behavioural difference, and no demonstrated link between them. Part of the
0.0121 may simply be noise at 15 passages.

**What would settle it:** rebuild one layer with RadixArk's *block* granularity (one `scale_2` per 128
experts) and re-measure NLL. If the gain disappears, granularity is the mechanism; if it survives,
something else is. That is a one-layer build and a re-probe — cheap, and it is the honest next step
rather than asserting the connection.

## The NLL advantage is real — 59 passages, and it has two components

Held-out set extended to **59 passages / 70,734 scored tokens over 11 languages**, fetched at offset
20000, disjoint from the calibration corpus *and* from the first held-out batch. Both arms scored on
identical text.

| | stock | v3 | Δ |
| --- | --- | --- | --- |
| overall NLL/token | 1.7379 | **1.7227** | **−0.0152** |

**Paired over passages: 40/59 favour v3, mean −0.0152, t = −3.92.** Not the 15-passage artefact I was
worried about — the effect grew with the sample.

### It splits into a general effect and a Devanagari effect

| subset | n | mean Δ | t |
| --- | --- | --- | --- |
| all | 59 | −0.0152 | −3.92 |
| **excluding Devanagari** | 53 | **−0.0069** | **−2.99** |
| Devanagari only | 6 | **−0.0891** | −19.08 |

The general improvement is real but small (~0.4 % relative). The Devanagari one is **13× larger** and
consistent across all six of its passages.

Per group: Devanagari −0.0891, **Thai −0.0249**, Cyrillic −0.0104, English −0.0096, Japanese −0.0059,
Arabic −0.0058, Greek −0.0045, Vietnamese −0.0022, Latin −0.0018, CJK −0.0003, Hebrew **+0.0021**.

### The ordering is the interesting part

**Devanagari and Thai lead, and those are precisely the two scripts the contract-broken builds
corrupted** (v1: Thai; v2: Thai + Devanagari). The same scripts that failed first under a bad export
gain most under a good one.

That suggests these scripts sit closest to the precision cliff in the expert weights — so any change
in expert quantization fidelity shows up there before anywhere else. It is a hypothesis, not a
result: consistent with everything measured, and not yet tested.

**The test that would settle it** is the one already identified: rebuild at RadixArk's *block*
`scale_2` granularity (3 values across 512 experts) instead of our per-expert one, and re-measure. If
the Devanagari gain collapses, granularity is the mechanism. Note that our own per-expert scales fit
each expert tighter, which is the only structural difference we have found and the only candidate
that predicts "helps most where precision is tightest".

## Granularity IS the mechanism — blk result, 2026-09-11 06:27

The block build (`--plain-max --scale2-block 128`) reproduces RadixArk's `weight_scale_2` granularity
— 3 distinct values across 512 experts — and changes nothing else.

| arm | granularity | overall NLL | vs stock |
| --- | --- | --- | --- |
| stock (base) | block | 1.7379 | — |
| **blk** | **block** | **1.7367** | **−0.0012 — inside noise** |
| lh_v3b | per-expert | 1.7227 | −0.0152 |

**Reproducing the base's granularity collapses the entire advantage**, and it collapses precisely
where the advantage lived: Devanagari **2.6298** (blk) against stock's 2.6431, versus **2.5541** for
per-expert. Thai likewise: blk 2.0614, stock 2.0592, per-expert 2.0343.

So the cause of our NLL gain over RadixArk is **per-expert `weight_scale_2`**, not Local-Hessian and
not the corpus.

### The full paired table

All three arms scored on the same 59 passages / 70,734 tokens, paired per passage.

| comparison | mean delta | paired t | passages favouring the first |
| --- | --- | --- | --- |
| `lh_v3b` vs stock | **−0.0152** | **−3.92** | 40/59 |
| `blk` vs stock | −0.0012 | −0.72 | 33/59 |
| `blk` vs `lh_v3b` | **+0.0140** | **+3.85** | 18/59 |

Split out, `blk` loses the advantage in both components, not just on average:

| | `lh_v3b` vs stock | `blk` vs stock |
| --- | --- | --- |
| excluding Devanagari (n=53) | −0.0069, t = −2.99 | **+0.0002, t = +0.12** |
| Devanagari (n=6) | −0.0891, t = −19.08 | −0.0133, t = −2.44 |
| Thai (n=6) | −0.0249, t = −1.72 | +0.0022, t = +0.60 |

The general gain goes to **exactly zero** at block granularity, and only 15 % of the Devanagari gain
survives. That is a collapse, not an attenuation.

### The confound that remains, stated

`blk` is plain-max + block; `lh_v3b` is Local-Hessian + per-expert. They differ in **two** things, so
strictly this shows "per-expert + LH beats block + plain max". The clean pair is `pmax` (per-expert +
plain max) against `blk` — and `pmax` was only measured on the older 15-passage set, where it scored
1.9542 against stock's 1.9663, i.e. most of the gain without any calibration.

### Correction: the base's blocks are uniformly 128, not 256/128/128

Censused `gate_proj.weight_scale_2` across layers 0, 1, 12, 24, 36, 47 of RadixArk's checkpoint. Every
layer has run lengths **[128, 128, 128, 128]** — one scale per block of 128 experts, 4 per layer.
Layer 24 reports *3* distinct only because two adjacent blocks happen to land on the same amax; the
blocking is identical. Earlier notes generalised layer 24's count and said "blocks of 256/128/128".
That was wrong, and it matters for anything we tell RadixArk, who would know their own blocking.

`blk` used `--scale2-block 128`, so it reproduced the real structure; the blk result stands.

### `pmax` re-measure in flight (started 06:40)

`lhbuild.py --plain-max --layers 0-47`, per-expert `weight_scale_2`, ~82 s/layer. Arm verified on
layer 0 before the run was trusted:

| | distinct gate `weight_scale_2` / 512 experts, layer 0 |
| --- | --- |
| stock (RadixArk) | **4** |
| `pmax` (this build) | 218 |
| `lh_v3b` | 228 |

218 rather than 512 is expected and not a defect: the amax comes off BF16 weights, so per-expert
maxima collide on the BF16 grid. gate/up mismatches 0/512; 6144 tensors.

**~~Re-measuring `pmax` on the 59-passage set is the one measurement still owed.~~ WRONG — it was
already measured.** Corrected 2026-09-11 07:00, after a rebuild was started and then killed. The
`pmax` arm was built *after* the contract fix, its 15 passages *include* Devanagari, and all three
arms were scored paired:

| 15-passage set | overall | paired t | Devanagari (n=2) |
| --- | --- | --- | --- |
| `pmax` (per-expert, no calibration) vs stock | −0.0121 | −1.52 | **−0.0792** |
| `lh_v3` (per-expert + Hessian) vs stock | −0.0175 | −2.15 | −0.0911 |
| `lh_v3` vs `pmax` | −0.0054 | **−1.18** | −0.0119 |

Plain max with **no calibration data at all** captures **69 % of the overall gain and 87 % of the
Devanagari gain**. Combined with `blk` — which reproduces the base's granularity and gains nothing
(+0.0002 excluding Devanagari, t = +0.12) — the 2×2 is complete without another build:

| | base's 128-blocks | per-expert |
| --- | --- | --- |
| plain max | `blk`: no gain | `pmax`: most of the gain |
| Local-Hessian | (never needed) | `lh_v3b`: the gain, +0.005 inside noise |

**Granularity is the mechanism. Local-Hessian's increment is not resolvable at either sample size.**

The cost of not re-reading this first: ~20 minutes of build and a deleted `blk` checkpoint. The
`check-field-before-expensive-steps` rule applies to our own results, not just the field. It separates:
granularity alone → 1.7227, or granularity plus a share from Local-Hessian. That distinction decides
what is worth telling RadixArk — "use per-expert scales" is actionable, "use per-expert scales *and*
Hessian calibration" much less so.

### Do we need a v4? No.

- **Weight-scale search is saturated.** `local_hessian` 8.585 %, `mse` 8.644 %, four-over-six 8.687 %,
  plain max 9.490 % — every method within **0.10 pp** of the others and all capturing ~0.85–0.90 pp
  of the available ~0.9. The residual is the 4-bit format, not the search.
- **v3 already has the mechanism**: per-expert `weight_scale_2`.
- **The only untouched lever is the activation scale**, and that needs a ModelOpt bump for
  `nvfp4_act_headroom` — not another build with current tooling.

## v3 is published — 2026-09-11 08:09

`josch15366/Qwen3.8-Flash-Next-NVFP4-LocalHessian-Experts`: 48/48 layer files, 63.32 GiB, verified
through the API rather than the uploader's own log (`layer00..layer47`, no zero-size entries).

**Two failed runs before it landed, both in HuggingFace's Xet CAS:**

| run | duration | error |
| --- | --- | --- |
| 06:07 | 90 min, 64.3/68 GB read | `ConnectionError` — `cas-server.xethub.hf.co/v1/xorbs/…` |
| 07:43 attempt 1 | 23 min | `TimeoutError: error decoding response body` |
| 07:43 attempt 2 | **1.7 min** | OK |

**The single-commit choice is what made the failures harmless.** `upload_folder` pushes the LFS
objects first and commits once, so after two crashes the public repo still showed 0 layer files — no
index referencing files that were not there. File-by-file upload would have left a broken checkpoint
public for 90 minutes.

**And the retry was worth more than the diagnosis.** After the second failure I had a concrete fix
ready (`HF_HUB_DISABLE_XET=1`, falling back to plain LFS multipart) and was one go-ahead from killing
the run to apply it. Attempt 2 then finished in **1.7 minutes** — Xet had kept the xorbs from the
23-minute attempt and only needed to commit. Switching would have discarded that and re-uploaded 63
GiB. Backoff first, rearchitecture second.

Card updated at publish time: the in-flight banner removed, and the stale "the cause has not been
identified" caveat replaced with the granularity result and its table.
