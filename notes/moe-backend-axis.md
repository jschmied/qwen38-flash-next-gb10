# The MoE kernel axis is closed while MTP is on

## Why we looked

Per-token bytes on `fp8head`, derived from the shard headers:

| group | per-token GiB |
| --- | --- |
| GDN (linear attn) | 1.95 |
| MoE experts (10 of 512) | 1.24 |
| hyper-connections | 1.19 |
| QSA (full attn) | 0.59 |
| lm_head | 0.59 |
| shared_expert | 0.44 |

At c=1 the experts are **20%** of the budget — but the dense side is read once per *step* and
shared across the batch, while each sequence routes to its own ten experts. The share crosses
50% at **c=4** and reaches **80% at c=16**. Every headline number we have published is c=1,
which is the regime where the experts matter least.

So: a better FP4 MoE kernel is the largest untouched lever *at load*.

## Three blockers, all real

**1. `--moe-backend` is global; this checkpoint is not.** `config.json`'s `ignore` contains
`mtp.*`, so `mtp.layers.0.mlp.experts.{gate_up,down}_proj` carry no scales (2 tensors, 4.86 GiB
BF16) while the body's 294,912 expert tensors are all scaled. One unquantized MoE layer in the
drafter vetoes the kernel choice for all 48 quantized layers:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
Expected one of ['triton','batched_triton','flashinfer_trtllm','flashinfer_cutlass','aiter'].
```

`map_unquantized_backend` (`fused_moe/oracle/unquantized.py:166`) raises rather than falling
back to a supported backend for the layers that cannot honour the request.

**2. It fails 10.5 minutes in.** Validation happens after weight load and torch.compile, not at
config parse. 17:51:56 engine init → 18:02:30 raise.

**3. With MTP off, b12x faults outright.** Illegal memory access during `profile_run`. It
surfaces at `hc.py:249 _hc_combine` inside Triton's `_init_handles`/`load_binary` — a kernel
*load*, not an execution, which is the signature of a context already poisoned by an earlier
async fault. `_hc_combine` is merely the next kernel to be JIT-loaded; the culprit is the b12x
MoE that ran before it.

## What this leaves

With MTP enabled the selectable set is `triton`, `batched_triton`, `flashinfer_trtllm`,
`flashinfer_cutlass`. `auto` already picks `FLASHINFER_CUTLASS`; `trtllm` is the known
silent-garbage path on sm_121 and `triton` is slower. **There is no move to make here.**

`CUTE_DSL_ARCH=sm_121a` was missing from `serve-fnext.sh` and has been added — it was required
for the b12x path on the 27B and had silently fallen off when the Flash-Next launcher was
written. It is harmless on the cutlass path.

## The lever this actually points at

Quantizing the **drafter's** MoE (`mtp.layers.0.mlp.experts.*`, 4.86 GiB BF16) would both free
~3.6 GiB resident and unlock the full backend set for the body. That is the prerequisite for
ever testing a native FP4 MoE kernel here, and it is a checkpoint edit of the kind we have done
four times already.

Not yet attempted.

## 2026-08-30 — de-risking the drafter-MoE build, before touching any weights

Four things established without writing a byte of new checkpoint.

**1. The config edit works, and touches nothing else.** Offline gate (~10 s, no GPU) via
`ModelOptMixedPrecisionConfig.from_config` + `_resolve_quant_algo`:

| probe | as shipped | after removing `mtp.*` from `exclude_modules` + adding a `quantized_layers` entry |
|---|---|---|
| `mtp.layers.0.mlp.experts` | **None** (unquantized) | **NVFP4** |
| `model.language_model.layers.0.mlp.experts` | NVFP4 | NVFP4 |
| `lm_head` | FP8_PB_WO | FP8_PB_WO |

**2. The drafter's quant config is resolved independently.** `get_draft_quant_config` reads the
*draft* model config (`model_executor/models/utils.py:883`), and `mtp.py:115-141` injects it. Since
our speculative config points at the same directory, one edit reaches both.

**3. Only one shard changes.** Both drafter expert tensors live in `model-bf16-00011.safetensors`
(8.08 GiB of 206 shards), so the build can hardlink 205 shards and rewrite one — which matters,
because the box has **21 GB free**.

**4. The layouts differ, and that is the work.** The body stores experts *per-expert, per-projection*
(`experts.{N}.{down,gate,up}_proj.weight` U8 + `weight_scale` + `weight_scale_2` + `input_scale`,
**6,144 tensors for layer 0 alone**). The drafter stores the *fused, stacked* form —
`gate_up_proj [512, 1280, 2560]`, `down_proj [512, 2560, 640]`, two BF16 tensors — which is what
`FusedMoE` holds internally as `w13`/`w2` (`mtp.py:358` maps
`"gate_up_proj": ["gate_proj", "up_proj"]`). The split is clean (gate = first 640 rows, up = last
640), so a per-expert emission is mechanical, if bulky.

### The fork that has to be decided before building

| | W4A16_NVFP4 (weight-only) | NVFP4 (W4A4) |
|---|---|---|
| `input_scale` | **not needed** | required |
| calibration | **none — data-free** | needs a calibration pass |
| memory | ~3.6 GiB freed | ~3.6 GiB freed |
| MoE backend | *"NVFP4 routed experts run via **Marlin W4A16**"* (`modelopt.py:2291`) | the scheme the CUTLASS/b12x kernels want |

So the cheap, data-free build most likely pins the MoE to **Marlin**, which is not the backend we
were trying to unlock — while the build that unlocks b12x is the one needing calibration, and
carries the trap that dropping `input_scale` while the algo stays `NVFP4` gives an uninitialised
activation scale and **zero characters of output, silently**.

**Status: the memory win is cheap and safe; the backend win is not the same build.** Unblocked and
ready to build either way, but the choice is a real one and worth making deliberately.
