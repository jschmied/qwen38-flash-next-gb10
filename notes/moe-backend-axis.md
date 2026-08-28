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
