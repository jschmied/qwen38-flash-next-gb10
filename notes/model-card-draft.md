<!-- Mirror of the card published PRIVATE at
     https://huggingface.co/josch15366/Qwen3.8-Flash-Next-NVFP4-LocalHessian-Experts-FP8Head
     (renamed from ...-W4A4-LocalHessian-Experts when the head was added to the shipping set)
     Keep this file and the repo README in step. mergeverify.py ships there too. -->

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

# Qwen3.8-Flash-Next — Local-Hessian NVFP4 experts + FP8 `lm_head`

**PRIVATE / WORK IN PROGRESS.** The build is running. Every `TBD` below is a cell the build must
fill. If a TBD cannot be filled, the claim it belongs to comes out rather than being softened.

Every measurement on this page is written up, with its method and its limits, in the open notes at
**[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)** — the
record of getting this model onto a single DGX Spark. Each claim below links to the note that
carries it. "We" throughout means that work; there is no single "our build", which is the point of
the next section.

## This is a partial checkpoint — two components

It contains the two things that were actually rebuilt:

| component | size | what it is | source |
| --- | --- | --- | --- |
| **routed experts** | **68.0 GiB** | NVFP4 W4A4, Local-Hessian calibrated | **this repo** |
| **`lm_head`** | **606 MiB** | blockwise FP8 (`F8_E4M3` + `weight_scale_inv`) | **this repo** |
| ple | 47.7 GiB | unchanged | RadixArk, bit-identical |
| attention | 5.1 GiB | unchanged | RadixArk, bit-identical |
| other dense + embed | 3.8 GiB | unchanged | RadixArk, bit-identical |

It is **not a servable model on its own.** Everything not listed as *this repo* comes unmodified from
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) and is
not republished here — 56.6 GiB of somebody else's unchanged weights carries no information. The exclusion set (`*.linear_attn.*`, `*.self_attn.*`, `*.ple.*`, `*.mlp.gate*`,
`lm_head`, embeddings, …) is RadixArk's and is unchanged.

## ⚠️ The FP8 `lm_head` needs a patched vLLM. The experts do not.

The two components in this repo have **different requirements**, and you can take either without the
other:

- **experts alone** — loads on stock vLLM. Nothing special.
- **`lm_head` too** — needs three local vLLM fixes, and is **TP=1 only**.

`lm_head` is BF16 `[248320, 2560]` in every published GPU checkpoint of this model, RadixArk's
included. That is not because quantizing it is unsafe — it is because three independent pieces of
vLLM plumbing prevent it, any one of which is sufficient on its own:

1. the model never passes `quant_config` to `ParallelLMHead`;
2. `config.json` is authoritative and its `ignore` list contains `lm_head`;
3. the vocab weight loader asserts `loaded_weight.shape[output_dim] == org_vocab_size`, which is
   false for the `[1940, 20]` block-scale companion — `VocabParallelEmbedding`'s loader has no
   concept of a scale tensor.

There are also **two** `lm_head` construction sites, not one: `mtp.py` builds its own
`ParallelLMHead`, also without `quant_config`, so with speculation enabled it fails identically with
`no module or parameter named 'lm_head.weight_scale_inv'`. Patch both.

TP=1 only, deliberately: our fix raises `NotImplementedError` above TP=1 rather than copying the
scale, because above TP=1 it needs sharding in *block* space (`rows // block_n`) and silently
mis-sharding it would produce wrong logits instead of an error.

Full write-up, including the scale-convention trap — ModelOpt's `weight_scale_inv` holds the scale,
**not** the reciprocal, and getting it backwards costs 565,100,324 % relative error instead of
2.2489 % — is at **[quantizing-lm-head.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md)**.

### What the head buys

**+11 % decode at no measurable quality cost.** NLL/token 0.9687 → 0.9628 over 646 held-out tokens of
prose, code, German, French and technical text; nine chunks improved and five worsened, so the sign
is mixed and this is reported as *no measurable cost*, not as an improvement. A bandwidth model
predicted ~+10 % from removing 0.64 GB/token; the measurement was +11 %.

Format matters more than bit-width here: on the sibling Qwen3.8-27B an **NVFP4** head measured 2.4 %
worse NLL in 8 of 8 chunks and was declined for production, while this **FP8** head is loss-neutral.

**One honest limit on that number:** it was measured *without speculation*, so the second
construction site never executed during validation — the configuration that validated the change was
not the configuration we serve. Re-validation under MTP is TBD, and until it is filled the +11 %
should be read as a no-speculation figure.

### It cannot affect the calibration

`lm_head` sits downstream of every MoE block, so it has no influence on the hidden states the experts
were calibrated on. The captured rows are valid for either head. It *does* matter for reading the
numbers below, which hold the head **constant across both arms** — otherwise two things vary at once.

## What is different about it

The per-group weight scales are chosen by **Hessian-weighted search** — ModelOpt's `local_hessian`,
the method of [arXiv 2608.28113](https://arxiv.org/abs/2608.28113) ("H-Scale", Qwen team) — instead
of the plain amax/MSE sweep used by the published builds.

**Same format, same tensor shapes, same file size, same inference speed.** Only the values of
`weight_scale` differ. This is a quality change at zero inference cost, not a speed or memory
optimisation.

Measured directly, on identical data, before the build: reconstruction error through the
Local-Hessian scales **8.18 %** against **9.53 %** for plain-max
([roadmap](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/modelopt-nvfp4-roadmap.md)).

## Calibration data

**SWE-bench Multilingual agent trajectories** — 18 instances, 398,861 tokens, spanning 15 projects
and roughly ten languages (Java, JS/TS, Go, Rust, Ruby, PHP, C/C++, Python, Lua, jq). Trajectories
were generated by Qwen3.8-27B; 78 % of the token mass is shell/tool output, which is the repos' own
text and model-independent.

The sample is **stratified by length**, not by project, because position inside the sequence is what
moves expert routing: rows from positions 0–8,063 and rows from beyond 8,063 share only **56 %** of
their top-50 expert set (routing total-variation 0.3747). An earlier capture that reached only
position 8,063 would have calibrated every expert on a distribution that does not hold across 64 %
of the model's real position range. This one reaches **95,239**.

Expert coverage at the layer measured: **0 experts with no routed rows, 0 thin (<64 rows)**, median
5,007 rows per expert.

Sources: [how the corpus was built](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/calibration-corpus.md) · [why it is stratified by length, and the measurement that forced it](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/modelopt-nvfp4-roadmap.md).

## Results

| | |
| --- | --- |
| layers rebuilt | TBD / 48 |
| experts per layer with a full Hessian | TBD |
| combining-mark regression (Thai/Devanagari/Arabic/Hebrew/ZWJ, vs stock) | TBD |
| SWE-bench Multilingual, held-out slice | TBD |

**On the SWE number, stated in advance:** resolution rate has a standard error of about 2.9 points at
300 instances, so it cannot resolve the 1–2 point differences at issue here. It is reported as a
**regression check** — evidence the rebuild did not break the model — and not as proof the
calibration helped. Calibration and evaluation instances are disjoint by construction: the 100
instances used to build the corpus are excluded from the evaluation pool.

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
| pure **expert** | 192 | 63.3 GiB | **drop from the index** — never referenced, never opened |
| pure **other**, no `lm_head` | 12 | 49.2 GiB | reference as-is |
| **mixed**: experts + others | 1 | 4.7 expert + 5.3 other | repack, keeping the non-expert tensors |
| **mixed**: `lm_head` + 169 others | 1 | 3.4 GiB | repack, keeping everything **except** `lm_head` |

So the merge writes about **7.6 GiB**. Everything else is the 12 untouched base shards, this repo's
48 layer files plus its `lm_head`, and a new index. No full copy, no repack of 58 GiB.

If you take the experts but **not** the head, the second row collapses back into "reference as-is"
and the merge writes 5.3 GiB.

Steps:

1. Fetch the base checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4`.
2. Classify its shards by reading headers only (8-byte length prefix + JSON).
3. Repack the two mixed shards: from the expert/other one keep the tensors **without**
   `.mlp.experts.` in the name; from the `lm_head` one keep everything **except** `lm_head.*`.
   (Skip the second if you are not taking the head.)
4. Build a new `weight_map`: expert tensors → this repo's `layerNN.safetensors`; `lm_head.*` → this
   repo's head file; every other tensor → its pure-other shard, or a repacked one.
5. Copy `config.json`, `hf_quant_config.json`, tokenizer files and the chat template from the base.
   **If you take the head, `config.json` is not unchanged**: its `ignore` list contains `lm_head`
   and is authoritative, so the head stays BF16 until you remove that entry — see the vLLM section
   above.
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

**Level 2 — per-tensor sha256 against the source each tensor should have come from.** Experts and
`lm_head` against this repo, everything else against the base. Proves provenance, not merely internal consistency.

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
