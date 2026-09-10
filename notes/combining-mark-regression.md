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
