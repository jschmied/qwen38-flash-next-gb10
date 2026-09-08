DRAFT — needs the user's go. vllm-project/vllm issue #55496 (2026-09-08).
We have NOT reproduced this: the checkpoint is not on our box. Everything below is source
inspection on our serving venv plus one confirmation. Say so in the post; do not imply a repro.

---

Confirming cause 1 on a newer build, and I think cause 2's "maintainer call" has a smaller answer
than it looks.

**Cause 1, still present on `0.28.1rc1.dev401+g8340fe1bb`** (newer than the preview you filed
against): `FP8_BLOCK_SCALES` occurs zero times in
`vllm/model_executor/layers/quantization/modelopt.py`, and `ModelOptMixedPrecisionConfig.get_quant_method`'s
`RoutedExperts` branch is still `FP8` / `NVFP4` / `W4A16_NVFP4` / `MXFP8` → `return None`. Nothing
to add to your analysis; just an independent second version.

**Cause 2 — the remap you need already exists in the plugin, applied to two fields and not to the
third.** `Qwen4ExpForCausalLM` resolves to `vllm.models.qwen4_exp`, which is out of tree, so a PR
against this repo cannot fix cause 2 at all — that may be the more useful half of this note.

In `vllm/models/qwen4_exp/nvidia/mtp.py`, `_remap_ignored_layers()` does exactly the shift you
describe:

```python
new_name = re.sub(r"(?<=\.layers\.)\d+",
                  lambda m: str(mtp_start_layer_idx + int(m.group(0))), name)
```

and `_make_draft_vllm_config()` applies it to the draft quant config's **`ignored_layers`** and
**`exclude_modules`** — but not to **`quantized_layers`**, which is the dict
`ModelOptMixedPrecisionConfig` actually dispatches on (`_resolve_quant_algo` reads it; the class
docstring calls it "the per-layer algorithm is specified in the `quantized_layers` dict"). `grep -rn
quantized_layers vllm/models/qwen4_exp/` returns nothing.

So this is not "mapper on the model class vs. quant config". Both exclusion fields are already
remapped in one place, in the plugin, and the field that drives per-layer dispatch was missed. The
minimal fix is a third `setattr` in `_make_draft_vllm_config` alongside the two that are there —
with one wrinkle: `quantized_layers` is a **dict keyed by layer name**, while `_remap_ignored_layers`
takes a list, so it needs a small key-mapping variant rather than a literal third call.

For completeness, a second route exists and is the #53790 pattern you cite:
`ModelOptMixedPrecisionConfig.apply_vllm_mapper` already does
`self.quantized_layers = hf_to_vllm_mapper.apply_dict(self.quantized_layers)`, so a mapper rule
carrying the index shift would also reach it. I have not checked whether `apply_vllm_mapper` runs on
the *draft* config, which is why I would take the `_make_draft_vllm_config` route — it matches how
the two neighbouring fields are already handled in that file.

Context for why we looked: on our GB10 we hit a related trap on this family of checkpoints —
MIXED_PRECISION dispatches off `quantized_layers`, not `config_groups`, and reading the wrong field
gives you a W4A4 kernel with no `input_scale`. Same field, different symptom.

We run a single GB10 at TP=1 with `RadixArk/…-NVFP4`, not the `nvidia/` checkpoint, so we cannot
reproduce your failure or confirm the acceptance-rate result. If a TP=1 single-node data point would
help a reviewer separate this from the TP=2 + EP arrangement, say so and we will fetch the
checkpoint.

<!-- AI disclosure: produced with AI assistance; I reviewed every line and ran every check quoted. -->
