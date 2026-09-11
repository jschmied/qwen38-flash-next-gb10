---
base_model:
- Qwen/Qwen3.8-Flash-Next
- RadixArk/Qwen3.8-Flash-Next-NVFP4
license: other
library_name: Model Optimizer
tags:
- ModelOpt
- NVFP4
- W4A4
- local-hessian
- partial-checkpoint
---

# Qwen3.8-Flash-Next — Local-Hessian NVFP4 experts


> 🗺️ **Part of the [Flash-Next Quant Map](https://claude.ai/code/artifact/3534a530-5e94-4ce2-abac-f1c70ee204e3)** — the measured landscape of
> Qwen3.8-Flash-Next quantization on a single DGX Spark: which schemes fit in 128 GB, what
> each costs in speed and quality, and where this piece sits among them.

> An earlier build was withdrawn after the combining-mark canary caught it corrupting Thai. The cause
> was two export-contract bugs of ours — `input_scale` written as `amax/6` instead of `amax/2688`, and
> gate/up not sharing one `weight_scale_2` — both now fixed and both covered by a verifier check that
> is exercised against the broken build so it is proven to fire. No weights were ever published. Full
> history in
> [combining-mark-regression.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/combining-mark-regression.md).

Every measurement on this page is written up, with its method and its limits, in the open notes at
**[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)** — the
record of getting this model onto a single DGX Spark. Each claim below links to the note that
carries it. "We" throughout means that work; there is no single "our build", which is the point of
the next section.

## This is a partial checkpoint — experts only

It contains **only** the routed expert tensors,
`model.language_model.layers.*.mlp.experts.<e>.{gate_proj,up_proj,down_proj}.*` — 63.3 GiB,
NVFP4 W4A4, Local-Hessian calibrated. **It runs on stock vLLM; no patches.**

It is not a servable model on its own. Everything else comes unmodified from
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) and is
not republished here — 57.9 GiB of somebody else's unchanged weights carries no information:

| component | size | source |
| --- | --- | --- |
| **routed experts** | **63.3 GiB** | **this repo** |
| ple | 47.7 GiB | RadixArk, bit-identical |
| attention | 5.1 GiB | RadixArk, bit-identical |
| MTP drafter experts | 4.7 GiB | RadixArk, bit-identical — BF16, `mtp.*` is excluded from quantization |
| other dense + embed + lm_head | 5.0 GiB | RadixArk, bit-identical |

The MTP drafter has its own expert pair and it is **not** part of this repo. Counting it with the
routed experts (a single "68.0 GiB experts" row) was an error in an earlier revision of this page.

The exclusion set (`*.linear_attn.*`, `*.self_attn.*`, `*.ple.*`, `*.mlp.gate*`, `mtp.*`, `lm_head`,
embeddings, …) is RadixArk's and is unchanged.

> **Also available, separately: a blockwise FP8 `lm_head`** — 606 MiB, **+11 % decode** at no
> measurable quality cost, usable on its own against any Flash-Next build. It is a **different repo**
> because it has different requirements: it needs a patched vLLM and TP=1, while these experts need
> neither. The two compose; neither depends on the other.
> → [Qwen3.8-Flash-Next-FP8-lm_head](https://huggingface.co/josch15366/Qwen3.8-Flash-Next-FP8-lm_head)

## `lm_head` does not affect any of this

Worth stating because the two are easy to conflate: `lm_head` sits downstream of every MoE block, so
whichever head you serve has no influence on the hidden states these experts were calibrated on. The
captured rows are valid for either. It *does* matter for reading the numbers below, which hold the
head **constant across both arms** — otherwise two things vary at once.

## What is different about it

> **Held-out NLL/token — 59 Wikipedia passages, 70,734 scored tokens, 11 languages**, fetched at
> offsets past the calibration corpus and never seen by any build:
>
> | | this build | base checkpoint |
> | --- | --- | --- |
> | NLL/token | **1.7227** | 1.7379 |
>
> **−0.0152, paired t = −3.92 over passages** (40/59 favour this build). It splits into a small
> general gain — **−0.0069, t = −2.99** with Devanagari excluded — and a large Devanagari-specific one,
> **−0.0891** across all six of its passages. By group: Devanagari −0.0891, Thai −0.0249, Cyrillic
> −0.0104, English −0.0096, then everything else under 0.006, with Hebrew the one small regression
> (+0.0021).
>
> **⚠️ The Hessian calibration is not why.** The cause is **`weight_scale_2` granularity**: we derive
> one per expert, the base derives one per block of **128 experts** (4 per layer, uniform across every
> layer censused). Rebuilding at the base's granularity and changing nothing else — plain max,
> `--scale2-block 128` — collapses the advantage:
>
> | vs base checkpoint | overall | excluding Devanagari (n=53) | Devanagari (n=6) |
> | --- | --- | --- | --- |
> | this build (per-expert) | **−0.0152**, t = −3.92 | −0.0069, t = −2.99 | −0.0891, t = −19.08 |
> | rebuilt at the base's granularity | −0.0012, t = −0.72 | **+0.0002, t = +0.12** | −0.0133, t = −2.44 |
>
> At block granularity the general gain is zero and 15 % of the Devanagari gain survives. A
> **plain-max** per-expert build — no calibration data at all — captured 69 % of the overall gain and
> 87 % of the Devanagari gain on a 15-passage set, beating Hessian calibration's increment by only
> 0.0054 (t = −1.18, not a result). **If you take one thing from this repo, take per-expert
> `weight_scale_2`, not the calibration method.**
>
> Both builds are also clean on the combining-mark canary (0/48 corrupt, 48/48 exact), matching the base.

**Same format, same tensor shapes, same file size, same inference speed.** Only the values of
`weight_scale` and `weight_scale_2` differ. This is a quality change at zero inference cost, not a
speed or memory optimisation.

## About the name: this build *is* Local-Hessian, but that is not where the gain comes from

The per-group weight scales are chosen by **Hessian-weighted search** — ModelOpt's `local_hessian`,
the method of [arXiv 2608.28113](https://arxiv.org/abs/2608.28113) ("H-Scale", Qwen team) — instead
of the plain amax/MSE sweep used by the published builds. That is what was built, and it is what the
repo is named after.

It does win on weight reconstruction: **8.585 %** against **9.494 %** for plain-max on identical
data, 0.91 pp. **But that advantage does not survive to held-out text.** Once `weight_scale_2`
granularity is held fixed, Local-Hessian's increment over plain max is 0.0054 NLL at t = −1.18 —
not a result. The 0.91 pp is real and it is nearly irrelevant.

**Treat reconstruction error with suspicion generally.** In this same work weight reconstruction was 0.004 pp between
our plain-max export and the base while two producer/runtime contracts were broken, and it rated an
earlier Local-Hessian build *better* than the base while that build corrupted 25 % of Thai copy
tasks. It measures the weights; it does not measure what the runtime does with them
([roadmap](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/modelopt-nvfp4-roadmap.md)).

## Calibration data

**Two halves, 221 records in total.**

| half | records | what it is |
| --- | --- | --- |
| **agent** | 100 | SWE-bench Multilingual trajectories — Java, JS/TS, Go, Rust, Ruby, PHP, C/C++, Python, Lua, jq. Generated by Qwen3.8-27B; ~78 % of the token mass is shell/tool output, i.e. the repos' own text and model-independent. |
| **script** | 121 | **Wikipedia articles** across **13 writing systems** — he, ja, th, ar, hi, zh, vi, de, fr, es, tr, ru, el. |

The script half exists because the first corpus was **99.94 % ASCII with zero Thai**, and a build
calibrated on it was the one the combining-mark canary caught. That turned out **not** to be the
cause — the cause was the two export-contract bugs named at the top of this page — but the corpus was
broadened anyway, and this build is the broadened one. Calibrating a 4-bit MoE only on ASCII is a bad
idea whether or not it was the bug.

From that corpus the capture fed **393,689 tokens across 67 items**, sampled **stratified by length
in three bands**, reaching position **95,239**. By tokens the mix is the opposite of the item counts,
because stratification pulls in a few very long agent runs:

| | items | tokens | share |
| --- | --- | --- | --- |
| agent trajectories | 12 | 273,747 | **69.5 %** |
| Wikipedia | 55 | 119,942 | 30.5 % |

One agent instance (`jqlang__jq-2650`) is 95,239 tokens on its own and is what reaches the far end of
the position range.

> **The held-out set is also Wikipedia, so disjointness was checked rather than assumed.** Comparing
> 8-gram shingles: **0 of the 21,653 held-out shingles appear among the calibration corpus's 67,002.**
> Not a title-level check — a text-level one, which is the one that matters when held-out passages are
> drawn from the same encyclopedia at different offsets.

**Why stratified by length rather than by project:** position inside the sequence is what moves
expert routing. Rows from positions 0–8,063 and rows from beyond 8,063 share only **56 %** of their
top-50 expert set (routing total-variation 0.3747), so an earlier capture that stopped at position
8,063 would have calibrated every expert on a distribution that does not hold across 64 % of the
model's real position range.

Expert coverage at the layer measured: **0 experts with no routed rows, 0 thin (<64 rows)**, median
5,007 rows per expert.

Sources: [how the corpus was built](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/calibration-corpus.md) · [why it is stratified by length, and the measurement that forced it](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/modelopt-nvfp4-roadmap.md).

## Results

| | |
| --- | --- |
| layers rebuilt | **48 / 48** |
| experts with a full Hessian (≥64 routed rows) | **24,485 / 24,576 — 99.63 %** |
| thin (1–63 rows) / no rows → plain max | 58 (0.24 %) / 33 (0.13 %) |
| layers at 512/512 | **32 / 48** |
| serves, and is genuinely different weights | **yes** — 95/96, 91/91, 75/75, 79/79 tokens diverge from stock across four prompts, max \|Δlogprob\| 0.63–1.46, both arms coherent with correct tool calls |
| combining-mark canary (Thai/Devanagari/Arabic/Hebrew/ZWJ, 6 reps) | **0/48 corrupt, 48/48 exact** — matches stock's 0/72, **and matches a plain-max build's 0/48** |
| SWE-bench Multilingual, held-out slice | not run — at n=10 its SE is ~15 points and it could not resolve what the canary catches in 35 minutes |

The imperfect 0.37 % concentrates at the ends and for different reasons: layer 0's routing is
degenerate (7 experts never fire), and layers 44–47 are the specialised tail (`lh 495–501`). Those
experts are the least-routed by construction, so they also fire least at serve time — measured on
*this* corpus, which is the caveat. Experts with no routed rows carry plain-max scales, i.e. exactly
what every published NVFP4 build applies to every expert; the floor is the standard method, not a
hole.

**"Genuinely different weights" is a claim we checked directly**, because nothing else would have:
sampling 18 (expert, matrix) pairs, 18/18 differ from the base, with 100 % of one expert's fp8 scale
bytes and 56.3 % of its packed weight bytes changed. An exporter that copied its input would pass
every other check here.

**Why there is no SWE-bench number:** resolution rate has a standard error of about 2.9 points at
300 instances and worse on any slice we could afford, so it cannot resolve the 1–2 point differences
at issue here. It was not run. Had it been, it would have been a regression check and not evidence
the calibration helped. The split is disjoint by construction either way — the 100 agent instances
above are drawn from SWE-bench Multilingual's 300, and the remaining 200 are the evaluation pool.

## Merging it with the base checkpoint

### ⚠️ Read this first: the failure mode is a model that loads cleanly and is silently the original

Safetensors shards are arbitrary containers — `model.safetensors.index.json` maps each tensor name to
a file, and nothing requires a layer's tensors to sit together. So merging is an **index rewrite**,
not a file operation, and the base checkpoint's packing does not have to match ours.

But rewriting the index **is not sufficient**, and the reason is easy to miss:

vLLM keeps only the files referenced in the index (`filter_duplicate_safetensors_files`) — then
iterates **every tensor in each kept file**, in `_natural_sort_key` order:

```
layer00.safetensors                 <- this repo, loads FIRST
model-00001-of-00131.safetensors    <- base checkpoint, loads AFTER
```

`l` sorts before `m`. If a referenced base shard still contains the old expert tensors, **they
overwrite the ones from this repo.** No error, no warning. You get a model that loads, runs, and is
bit-for-bit the original quantization — and a merge that looks like it worked.

Do not "fix" this by renaming files so they sort last. Correctness must not depend on filename
collation. The old expert bytes have to be **absent from every referenced file**.

([full write-up, including the shard census and both verifier controls](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/modelopt-nvfp4-roadmap.md))

### The merge

Fortunately the base checkpoint is already almost perfectly separated by component. Of its 206
shards:

| shard class | count | size | what to do |
| --- | --- | --- | --- |
| holds tensors **this repo replaces** | 192 | 63.3 GiB | **drop from the index** — never referenced, never opened |
| everything else | 14 | 57.9 GiB | reference as-is |
| mixed | **0** | — | — |

**Zero mixed shards, so the merge writes no bytes at all** — 192 shards dropped, 14 linked, one new
index, `config.json` unchanged. No copy, no repack.

> ⚠️ **Classify by tensor name against *this repo's* contents, never by the pattern
> `.mlp.experts.`** — that pattern also matches `mtp.layers.0.mlp.experts.*`, the **MTP draft
> module's own experts**, which this repo does **not** replace. Dropping their shard leaves the
> checkpoint missing them and MTP will not load. We hit exactly this: the merge came out at 296,473
> of 296,475 tensors. Read the names out of this repo's files and treat everything else as
> "keep" — which is also why there are no mixed shards.

Steps:

1. Fetch the base checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
2. Read the tensor names out of this repo's `layerNN.safetensors` files — that set, and only that
   set, is what you replace. Then classify the base's shards by headers only (8-byte length prefix
   + JSON) against it.
3. If any shard turns out to be mixed, repack it keeping only the tensors this repo does **not**
   replace. Against the RadixArk base there are none.
4. Build a new `weight_map`: expert tensors → this repo's `layerNN.safetensors`; every other tensor →
   its pure-other shard, or the repacked one.
5. Copy `config.json`, `hf_quant_config.json`, tokenizer files and the chat template from the base,
   unchanged.
6. **Verify before serving** (next section). The merge is not done until it passes.

### Verifying the merge

`mergeverify.py` ships in this repo. A per-file checksum is the wrong instrument — in the failure
mode every file is individually valid.

**Level 1 — name collisions and index completeness.** Headers only, seconds. Every referenced file's
tensor names must be pairwise disjoint and their union must equal the index exactly. This is the check
that catches the trap above.

```
python mergeverify.py --merged <dir> --level1-only
```

It is exercised in both directions rather than assumed: it reports PASS on an untouched base
checkpoint (296,475 tensors across 206 files, no collisions) and FAIL, naming the tensor and both
files, on a synthesised collision.

**Level 2 — per-tensor sha256 against the source each tensor should have come from.** Experts against
this repo, everything else against the base. Proves provenance, not merely internal consistency.

```
python mergeverify.py --merged <dir> --ours <this repo> --stock <base>   # --sample 0 hashes all
```

**Level 3 — runtime divergence against the base.** The only check a correct-looking directory cannot
fake, and the reason levels 1 and 2 are not sufficient alone. Serve both checkpoints and compare
logprobs on a fixed prompt. Two conditions, both required:

- divergence from the base must be **non-zero**. Identical logprobs mean the merge silently fell back
  to the original, whatever the files say.
- output must still be coherent — which rules out the opposite failure, a merge that differs because
  it is broken.

A byte-level readback of a loaded weight is *not* a usable substitute: the loader may repack or
interleave quantized weights for the kernel, so a hash mismatch would not distinguish a bad merge
from a legitimate layout transform.

## Provenance and credit

Base model [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). All
non-expert weights and the quantization exclusion set are
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)'s
work, unmodified. Calibration method is the Qwen team's H-Scale as implemented in NVIDIA
ModelOpt 0.46.0. Built and measured on a single DGX Spark (GB10, sm_121, 128 GB unified memory).
