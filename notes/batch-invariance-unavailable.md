# vLLM's batch-invariance switch does not apply to this model

`VLLM_BATCH_INVARIANT=1` is vLLM's documented remedy for run-to-run nondeterminism.
On Flash-Next it does not start at all:

```
RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN.
```

Raised at `v1/attention/selector.py:240-244`, in `_cached_get_mamba_attn_backend`,
gated on `mamba_attn_backend.supports_batch_invariance()`.

## Why it is structural, not a misconfiguration

`AttentionBackend.supports_batch_invariance()` defaults to **False** (`v1/attention/backend.py:304`).
Exactly five backends override it to True, and all five are full-attention:

| backend | file |
| --- | --- |
| `triton_attn` | `backends/triton_attn.py:333` |
| `flash_attn` | `backends/flash_attn.py:153` |
| `flex_attention` | `backends/flex_attention.py:118` |
| `flashattn_mla` | `backends/mla/flashattn_mla.py:68` |
| `triton_mla` | `backends/mla/triton_mla.py:137` |

**No mamba / linear-attention backend implements it.** Batch invariance was built for the
full-attention family and never extended to the linear-attention one.

Flash-Next is a hybrid: `layer_types` is **36 `linear_attention` + 12 `full_attention`** of 48.
Three quarters of the stack runs the GDN path, so there is no combination of flags that gets
this model into batch-invariant mode. Note also that **FlashInfer is not on the list either**,
so the 12 full-attention layers would not be covered even if the GDN layers were — the mamba
selector simply raises first.

The same holds for the MoE path: `modular_kernel.py:587` gates on `_supports_batch_invariance()`,
which only `cutlass_moe`, `triton_moe` and `fused_humming_moe` implement.

## Consequence for the determinism investigation

The one off-the-shelf remedy is unavailable, so it cannot be used as a control arm and cannot be
offered as a workaround to anyone running this architecture. It also means the usual reasoning
("if batch invariance fixes it, it is batch-dependent kernel selection") is not testable here;
that hypothesis has to be attacked by other means.

It does **not** by itself explain our divergence. Batch invariance addresses variation caused by
differing batch composition; our probe diverges across three sequential identical requests at
concurrency 1, where batch composition is already constant.

Measured 2026-09-01, vLLM `0.1.dev20073+g8e685d198`.
