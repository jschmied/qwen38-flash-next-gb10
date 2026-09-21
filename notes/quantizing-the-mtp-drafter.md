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
4. **vLLM reads `config.json["quantization_config"]`, NOT `hf_quant_config.json`.** Both files
   exist here and both carry a full config. Editing only the latter leaves the drafter unquantized
   and the load dies with
   `AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_input_scale'`.
   Also: the key must be the **runtime** prefix — vLLM remaps the drafter to
   `layers.<num_hidden_layers>` = 48 — and the value must be a **dict**
   `{"quant_algo": "NVFP4", "group_size": 16}`, because `modelopt.py` does
   `quantized_layers[k]["quant_algo"]`. `_quantized_layer_prefix_candidates()` only swaps
   `language_model.model.` <-> `model.language_model.`, so `mtp.layers.0...` never matches.

## Status

Weights verified against the body's format (U8 `(640,1280)` / `(2560,320)`, scales
`float8_e4m3fn` `(640,160)` / `(2560,40)`, index fully resolving). **Whether vLLM loads a
split-per-expert NVFP4 MTP is still unproven** — `_resolve_quant_algo` strategy 3 exists for this
prefix shape but no published checkpoint exercises it. Acceptance length is unmeasured.
