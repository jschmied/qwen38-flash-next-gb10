<!-- Mirror of the card published PUBLIC at
     https://huggingface.co/josch15366/Qwen3.8-Flash-Next-FP8-lm_head -->

---
base_model:
- Qwen/Qwen3.8-Flash-Next
license: other
library_name: Model Optimizer
tags:
- ModelOpt
- FP8
- lm_head
- partial-checkpoint
---

# Qwen3.8-Flash-Next — blockwise FP8 `lm_head`

> 🗺️ **Part of the [Flash-Next Quant Map](https://claude.ai/code/artifact/3534a530-5e94-4ce2-abac-f1c70ee204e3)** — the measured landscape of
> Qwen3.8-Flash-Next quantization on a single DGX Spark: which schemes fit in 128 GB, what
> each costs in speed and quality, and where this piece sits among them.


A single tensor and its scale: `lm_head` quantized to blockwise FP8. **606 MiB, +11 % decode, no
measurable quality cost.** It drops into an otherwise untouched Qwen3.8-Flash-Next checkpoint.

| file | dtype | shape | size |
| --- | --- | --- | --- |
| `lm_head.weight` | `F8_E4M3` | (248320, 2560) | 606.2 MiB |
| `lm_head.weight_scale_inv` | `F32` | (1940, 20) | 0.1 MiB |

Measurements and method are in the open notes at
**[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)**; this
tensor's write-up is
[quantizing-lm-head.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/quantizing-lm-head.md).

## Why this does not already exist

`lm_head` is BF16 `[248320, 2560]` in **every** published GPU checkpoint of this model — verified
across ten of them. Only the MLX/Apple tier quantizes it. That is not because it is unsafe. It is
because three independent pieces of vLLM plumbing prevent it, any one of which is sufficient alone:

1. the model never passes `quant_config` to `ParallelLMHead`;
2. `config.json` is authoritative and its `ignore` list contains `lm_head`;
3. the vocab weight loader asserts `loaded_weight.shape[output_dim] == org_vocab_size`, which is true
   of `weight` and false of its `[1940, 20]` block-scale companion — `VocabParallelEmbedding`'s
   loader has no concept of a scale tensor.

**So this needs a patched vLLM.** There is no configuration flag that makes stock vLLM load it.

There are also **two** `lm_head` construction sites, not one: `mtp.py` builds its own
`ParallelLMHead`, also without `quant_config`, so with speculation enabled it fails identically with
`no module or parameter named 'lm_head.weight_scale_inv'`. Patch both.

**TP=1 only, deliberately.** Our fix raises `NotImplementedError` above TP=1 rather than copying the
scale: above TP=1 it needs sharding in *block* space (`rows // block_n`), and silently mis-sharding
it would produce wrong logits instead of an error.

## What it buys

| | BF16 head | FP8 head | Δ |
| --- | ---: | ---: | ---: |
| code | 23.2 tok/s | **26.1** | +12.5 % |
| factual | 23.7 tok/s | **26.2** | +10.5 % |
| German | 23.5 tok/s | **25.4** | +8.1 % |
| NLL/token | 0.9687 | **0.9628** | −0.60 % |
| tasks passed | 10/10 | 10/10 | — |

Paired NLL over 14 chunks / 646 tokens of held-out prose, code, German, French and technical text.
Nine chunks improved, five worsened — **mixed signs, so this is noise, not damage.** Reported as
*no measurable cost*, not as an improvement. A bandwidth model predicted ~+10 % from removing
0.64 GB/token; the measurement was +11 %.

**Format decides this layer, not bit-width.** On the sibling Qwen3.8-27B an **NVFP4** head measured
2.4 % worse NLL in 8 of 8 chunks and was declined for production, while this **FP8** head is
loss-neutral.

### With speculation it is worth considerably more

The +11 % above is the **no-speculation** figure. Under MTP (k=2) the head is worth more, not less,
because of *where* it sits in the speculative loop:

| | c=1 | c=16 |
| --- | ---: | ---: |
| BF16 head + MTP | 29.2 tok/s | 139.6 |
| **FP8 head + MTP** | **36.3** | **167.8** |
| | **+24 %** | **+20 %** |

It also flips speculation from a **net loss** at c=16 (139.6 against 156.0 without MTP) to a **net
gain** (167.8).

The mechanism: MTP evaluates `lm_head` once per draft token *plus* once for verification, so at k=2
the head is on the critical path about **three times per step**, while dense projections are read
once per step either way. Halving the head therefore pays k+1 times over.

**Acceptance is unchanged** — mean accepted length 2.21 against 2.15 — so this is pure speed, not
better drafting. If you were expecting a quantized layer to cost the drafter accuracy, it did not.

The general rule this is one instance of: **a quantization lever composes with speculation when the
layer is evaluated per draft token, and competes with it when the layer is read per step.** Worth
checking before assuming either — on this same box, quantizing the dense projections *reduced* MTP's
benefit from +67 % to +23 % and turned it into a net loss past c≈4.

**One caveat that does stand:** `mtp.py` builds its own `ParallelLMHead` and must be patched too, or
speculation fails outright with `no module or parameter named 'lm_head.weight_scale_inv'`.

## Compatibility — check, do not assume

This head was quantized from one specific BF16 `lm_head`. A checkpoint shipping a different one will
produce **wrong logits, not an error.**

Reference sha256 and a ready-made checker are in **[COMPAT.md](COMPAT.md)**. The tensor is
`BF16 (248320, 2560)`, sha256
`40bddd25d0d94a128ab08280faad39cfc3ee3064252761269115f544722607c9` — verified 2026-09-10 to be
bit-identical in Qwen's BF16 parent and in RadixArk's NVFP4 build, which ships it unchanged.

That basis is **two checkpoints, on one date**, not a survey of the field. Run the check rather than
trusting the list.

## Using it

1. Take any Qwen3.8-Flash-Next checkpoint whose `lm_head` matches the reference hash.
2. Repoint `lm_head.weight` and `lm_head.weight_scale_inv` in its `model.safetensors.index.json` at
   this repo's file, and make sure no other **referenced** shard still contains an `lm_head` tensor —
   a duplicate name in two referenced files means the later one wins by `_natural_sort_key`, with no
   error. In the RadixArk build `lm_head` shares a shard with 169 other tensors, so that shard has to
   be repacked without it (~2.3 GiB written).
3. Remove `lm_head` from `config.json`'s `ignore` list — it is authoritative, and the head stays BF16
   until you do.
4. Serve on a vLLM carrying the three fixes above, TP=1.

## Related

A Local-Hessian NVFP4 rebuild of the **routed experts** for the same model is in progress as a
separate release, with different requirements — it needs no vLLM patches, and the two compose without
either depending on the other. Not published yet; it will be linked here when it is. Its method and
results are being written up in the notes repo linked above.

Base model [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Built and
measured on a single DGX Spark (GB10, sm_121, 128 GB unified memory).
