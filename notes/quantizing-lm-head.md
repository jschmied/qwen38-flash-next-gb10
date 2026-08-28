# Quantizing `lm_head`: +11% for nothing, and why nobody had done it

`lm_head` is BF16 `[248320, 2560]` in **every** published GPU checkpoint of Qwen3.8-Flash-Next —
verified across ten of them. Only the MLX/Apple tier quantizes it. That is not because it is
unsafe. It is because three independent pieces of vLLM plumbing prevent it, and any one of them
alone is sufficient.

We quantized it to blockwise FP8 and measured **+11% decode at no measurable quality cost.**

## Result

| | `lm_head` BF16 | `lm_head` FP8 | Δ |
|---|---:|---:|---:|
| code | 23.2 tok/s | **26.1** | +12.5% |
| factual | 23.7 tok/s | **26.2** | +10.5% |
| German | 23.5 tok/s | **25.4** | +8.1% |
| NLL/token | 0.9687 | **0.9628** | **−0.60%** |
| tasks | 10/10 | 10/10 | — |

Paired NLL over 14 chunks / 646 tokens of held-out prose, code, German, French and technical text.
Nine chunks improved, five worsened — **mixed signs, so this is noise, not damage.** We report it
as *no measurable cost*, not as an improvement.

The bandwidth model predicted ~+10% from removing 0.64 GB/token; we measured +11%.

**Why FP8 and not NVFP4.** On the sibling Qwen3.8-27B we previously measured an **NVFP4** head at
2.4% worse NLL (8 of 8 chunks) and declined it for production, while the **FP8** head we run in
prod is loss-neutral. The layer is precision-sensitive; the *format* is what decides it.

## The three blockers

### 1. The model never passes `quant_config` to `ParallelLMHead`

`vllm/models/qwen3_8_flash_next/nvidia/model.py`:

```python
self.lm_head = ParallelLMHead(
    config.vocab_size,
    config.hidden_size,
    prefix=maybe_prefix(prefix, "lm_head"),   # no quant_config
)
```

`VocabParallelEmbedding.__init__` then sees `quant_config is None` and installs
`UnquantizedEmbeddingMethod` unconditionally. The checkpoint's declaration is never consulted —
`get_quant_method` is not called at all. Fix: pass `quant_config=self.quant_config`.

### 2. `config.json` is authoritative, and its `ignore` list contains `lm_head`

`ModelOptMixedPrecisionConfig` reads the `quantized_layers` dict inside **`config.json`'s
`quantization_config` (preferred)**, falling back to `hf_quant_config.json` only as legacy. The
exclusion also lives under the `ignore` key, not `exclude_modules`:

```
ignore: [..., 'model.visual.*', 'model.language_model.embed_tokens', 'lm_head', ...]
```

So editing `hf_quant_config.json` — the obvious file, and the one whose name says "quant config" —
changes nothing. **The field you read for placement is the field you must write.**

### 3. The vocab weight loader rejects non-vocab-shaped tensors

Even with 1 and 2 fixed, loading dies:

```
assert loaded_weight.shape[output_dim] == self.org_vocab_size
```

`ParallelLMHead` routes every parameter through `VocabParallelEmbedding`'s vocab-aware loader,
which is true of `weight` `[248320, 2560]` and false of its block-scale companion `[1940, 20]`.
The embedding loader has no concept of a scale tensor.

Our fix attaches a plain copy loader to the scale parameters, **with an explicit
`NotImplementedError` for TP>1** — at TP=1 the vocab dimension is not sharded so a direct copy is
correct, but above that the scale needs sharding in *block* space (`rows // block_n`), and
silently mis-sharding it would yield wrong logits rather than an error.

## There are two `lm_head` construction sites, not one

`mtp.py` builds its own `ParallelLMHead`, also without `quant_config`. With speculation enabled the
MTP module consumes the same quantized tensors and fails identically:

```
ValueError: There is no module or parameter named 'lm_head.weight_scale_inv'
            in Qwen3_8FlashNextMTP
```

Worth stating plainly how we found it: we measured the `lm_head` result **without speculation**, so
the second site never ran. The configuration that validated the change was not the configuration we
serve. If you patch this, patch both — and re-run the validation under the flags you actually use.

## Doing it yourself

`scripts/quant_lmhead.py` builds the variant. Two things to get right:

**The scale convention.** ModelOpt `FP8_PB_WO` stores `weight_scale_inv`, which despite the name
holds the **scale, not the reciprocal**. Determine it empirically rather than assuming — take a
tensor the checkpoint quantized and a BF16 original of the same tensor from the parent checkpoint:

```
w_fp8 * scale  ->  2.2489% mean relative error
w_fp8 / scale  ->  565,100,324%
```

Our quantizer's own round-trip lands at **2.2502%** against the upstream author's 2.25% on their
layers — an independent implementation agreeing to four digits is the cross-check worth having.

**`config.json` must be copied, not hardlinked.** Building a variant by hardlinking the unchanged
files is the right move (it costs zero disk), but `config.json` must be excluded from that: an
edit through a shared inode silently rewrites the *parent* checkpoint. We did this and had to
restore the source.

## Gate it offline first

This cost three ten-minute boot cycles that a ten-second offline check would have prevented:

```python
from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
c = ModelOptMixedPrecisionConfig.from_config(
        json.load(open("config.json"))["quantization_config"])
c.is_layer_excluded("language_model.lm_head")   # must be False
c._resolve_quant_algo("language_model.lm_head") # must be the algo you wrote
```

Note the prefix: vLLM asks about **`language_model.lm_head`**, not `lm_head`.

## Open

Resident memory went **up**, 74.13 → 76.5 GiB, when removing 1.27 GB of BF16 should have taken it
down ~0.6. Run-to-run variance on this figure is ~0.2 GiB, so the ~2.4 GiB is real and
unexplained. The leading hypothesis is that a dequantized BF16 copy is retained alongside the FP8
weight — if so, the memory saving is still on the table.
