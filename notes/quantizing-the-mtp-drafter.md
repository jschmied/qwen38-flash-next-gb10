# Quantizing the MTP drafter to NVFP4

Built 2026-09-21. Script: `scripts/quant_mtp_nvfp4.py`. Output:
`/opt/llm/models/qwen38-flash-next-mtpfp4`.

## Why

The MTP head is **5.21 GB and 96.6% of it is its MoE experts**, shipped BF16 in every published
checkpoint (`mtp.*` sits in `exclude_modules`). Two costs:

- ~2.5-3.6 GB of resident memory that could be KV cache;
- vLLM routes the drafter through `UnquantizedFusedMoEMethod`, whose backend list is
  `['triton','batched_triton','flashinfer_trtllm','flashinfer_cutlass','aiter']`. Any
  quantization-only `--moe-backend` then aborts at drafter construction — this is why
  `flashinfer_b12x` + MTP could not start (see `moe-backend-axis.md`).

A quantized drafter removes the second problem at the source, with no vLLM patch.

## The 27B recipe does not transfer

`bench/nvfp4-table/data/fp8-drafter/quantize-dflash2-fp8.py` (the published
`josch15366/Qwen3.8-27B-DFlash2-FP8`) quantizes **2-D** tensors only. On Flash-Next that is 0.18 GB
of 5.21 GB. The experts here are fused and 3-D:

| tensor | shape | size |
|---|---|---|
| `mtp.layers.0.mlp.experts.gate_up_proj` | `[512, 1280, 2560]` | 3355 MB |
| `mtp.layers.0.mlp.experts.down_proj` | `[512, 2560, 640]` | 1678 MB |
| all 2-D BF16 weights combined | | 181 MB |

So the script splits each fused tensor into the **body's** per-expert 2-D NVFP4 layout, which is
the layout vLLM's NVFP4 MoE loader knows:
`experts.{i}.{gate,up,down}_proj.{weight,weight_scale,weight_scale_2,input_scale}`.

Left BF16 on purpose: norms, gates, and the hyper-connections — quantizing those is a closed lever
(`why-the-hyper-connections-do-not-respond.md`).

Result: 1536 expert projections, rewritten shards **11.54 -> 7.92 GB (-31.3%)**. Only 3 of 206
shards contain mtp tensors; the other 203 are hardlinked, so the checkpoint costs ~8 GB of disk.

## Four traps, each cost one 11-minute load

1. **Build ran as root** -> shards `0600 root:root`; the engine runs as `llm` and cannot read them.
   `chown -R llm:llm` — and note `chown -R` follows hardlinks into the source checkpoint, so check
   inode sharing afterwards (the rewritten shard must NOT share an inode; the other 203 must).
2. **`weight_scale` dtype.** `fp4_quantize` returns block scales as raw bytes; the format wants
   `float8_e4m3fn`, as the body stores them. `sf.view(torch.float8_e4m3fn)`.
3. **`input_scale` is a CALIBRATED activation scale, not 1.0.** The body carries e.g. `0.00202288`.
   An uncalibrated one yields "no error, plausible-looking output, wrong"
   (`choosing-a-quant-scheme.md`). We have no calibration set for the drafter, so we **borrow the
   body's last-layer per-expert values** — the MTP head consumes the final hidden state, so those
   statistics are the closest available. **This is an approximation; acceptance length is the
   measurement that exposes it** (`scripts/accept_probe.py`).
4. **There are TWO exclusion lists, and the one that matters is `ignore`.**
   `config.json.quantization_config` carries both `exclude_modules` (here: `[]`) and a
   compressed-tensors-style **`ignore`** list, which is merged into `exclude_modules` at
   construction. `ignore[9] = "mtp.*"`, `ignore[10] = "model.mtp.*"`. Clearing only
   `exclude_modules` leaves the drafter excluded, so it loads unquantized and dies on the extra
   tensors. Note a wildcard entry does not show up in a naive `"mtp.*" in prefix` substring test —
   the match is by fnmatch, which is how I convinced myself twice that nothing was excluded.
5. **vLLM reads `config.json["quantization_config"]`, NOT `hf_quant_config.json`.** Both files
   exist here and both carry a full config. Editing only the latter leaves the drafter unquantized
   and the load dies with
   `AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_input_scale'`.
   Also: the key must be the **runtime** prefix — vLLM remaps the drafter to
   `layers.<num_hidden_layers>` = 48 — and the value must be a **dict**
   `{"quant_algo": "NVFP4", "group_size": 16}`, because `modelopt.py` does
   `quantized_layers[k]["quant_algo"]`. `_quantized_layer_prefix_candidates()` only swaps
   `language_model.model.` <-> `model.language_model.`, so `mtp.layers.0...` never matches.

## Status: IT LOADS

`SERVES after 705s`, coherent generation, and — scoped to that boot — **zero
`Unquantized MoE backend` lines**, only the two expected `FLASHINFER_CUTLASS NvFp4` (worker + PLE
worker). So a split-per-expert NVFP4 MTP does load: `_resolve_quant_algo` strategy 3 (prefix match
on `.experts`) handles it, which no published checkpoint had exercised.

That removes the structural problem at its source: the drafter no longer takes
`UnquantizedFusedMoEMethod`, so a quantization-only `--moe-backend` should no longer abort. If that
holds, the generalized #56964 patch is not needed in OUR venv (it is still needed upstream, for
people whose drafter stays BF16).

**Still unmeasured: acceptance length**, which is the only thing that exposes the borrowed
`input_scale`. A/B against the BF16 drafter is running (`scripts/accept_probe.py`).

**Do not quote a memory win from single boots.** Today's KV figures (1,044,206 / 910,950 / 880,366 /
992,870 / 610,889 tokens) span different `FN_UTIL` and `FN_EXTRA` settings and are not comparable;
the A/B holds those fixed.

## The debugging loop that worked

Three of the four failed loads were config plumbing, diagnosed by guessing and costing ~11 minutes
each. The fourth question was answered in 30 seconds by instantiating the config object directly:

```python
cfg = ModelOptMixedPrecisionConfig.from_config(json.load(open("config.json"))["quantization_config"])
cfg._resolve_quant_algo("mtp.layers.48.mlp.experts")   # -> 'NVFP4'
cfg.is_layer_excluded("mtp.layers.48.mlp.experts")     # -> True  <- the actual bug
```

Test the predicate offline before paying for a load.

## Finding: the quantized drafter replaces the #56964 patch (2026-09-22)

`b12x + MTP n=3 + the NVFP4 drafter`, with `oracle/unquantized.py` reverted to **stock**
(`GENFIX56964` markers = 0) and only the #57946 padding fix left installed:

```
verdict: SERVES after 675s
nvfp4.py:261]  Using 'FLASHINFER_B12X' NvFp4 MoE backend
SpecDecoding metrics: Mean acceptance length: 2.93
```

So the drafter never reaches `UnquantizedFusedMoEMethod`, and the generalized #56964 fall-through is
**not needed in our venv**. Prod can carry one local patch (#57946, already upstreamed) instead of
two. Upstream still needs #56964's generalized form for everyone whose drafter stays BF16 — which is
all ten published Flash-Next variants, since every one excludes `mtp.*`.

Verification note: the `nvfp4.py:261` line sits just outside a 23:55 journal window and a first grep
missed it, which briefly looked like an unverified fallback. Widening the window found it. Scope
journal greps generously; "the line is absent" is a claim about the window, not the run.

## Open

`mtp-kvgain-spec.json` — does the ~3.6 GB freed from the drafter become usable KV? The speed A/B
(finding 194) pinned KV at 2 GiB and could not see it. Staged, not run.

## The draft and target share ONE lm_head (2026-09-22)

Checked because the idea came up of giving the drafter a 4-bit head while the body keeps FP8.

`Qwen4ExpMTP` does construct its own `ParallelLMHead` (`mtp.py:384`,
`prefix=maybe_prefix(prefix, "lm_head")`), and `tie_word_embeddings` is `False`. But
`load_eagle_model` then **throws that copy away**
(`v1/worker/gpu/spec_decode/eagle/utils.py:111-127`):

```python
if target_lm_head is not None and _should_share(eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head):
    if draft_lm_head is not None:
        del eagle_model.lm_head
    eagle_model.lm_head = target_lm_head
    # per-layer shared_head.head copies are deleted too
```

`_should_share` returns `True` as soon as the model does not set `has_own_lm_head`, and Qwen4Exp does
not set it. So **there is no second head in memory** — an earlier claim in this session that there
were two 636 MB copies was wrong.

Consequences for a distinct 4-bit draft head:

- it **costs** ~318 MB (a new NVFP4 copy), it does not free 318 MB;
- it needs a vLLM change: set `has_own_lm_head` on `Qwen4ExpMTP` and give the draft head a
  distinguishable prefix. The resolver needs no change — `_quantized_layer_prefix_candidates()` tries
  the full prefix first, so an explicit `quantized_layers["mtp.lm_head"]` entry beats the `lm_head`
  fallback it appends for anything ending in `.lm_head`;
- the payoff is drafting bandwidth, not capacity. `lm_head` is 636 MB at FP8 = ~2.3 ms per read at
  273 GB/s, read ~4x per step at MTP n=3 against a ~25 ms step. Consistent with our own BF16->FP8
  head result (+11% decode). A 4-bit draft head plausibly buys ~10% decode for ~1.3% of KV.
- body logits are untouched, and both axes are already measurable here: acceptance for draft quality,
  tok/s for the gain.

Not built. Recorded so the arithmetic is not re-derived.

