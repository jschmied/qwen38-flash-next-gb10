# The model and the memory budget

<!-- moved out of README.md on 2026-09-05 so the README stays an entry point; content unchanged -->

## The problem in one table

| build | weights |
|---|---|
| `Qwen/Qwen3.8-Flash-Next` (BF16) | 335.3 GiB |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 172.8 GiB |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | **125.9 GiB** |
| **usable on this box** | **~117 GiB** |

It splits cleanly, which is what makes the attempt work: main model 78.2 GiB (196 files), PLE
n-gram table 47.7 GiB (10 files, already FP8). Everything hinges on keeping the n-gram table out
of resident memory — and **the PLE offload is not the bottleneck**: major faults per token *fall
4.4×* from c=1 to c=48, because batched tokens share n-gram rows.

## What the model is

`Qwen4ExpForConditionalGeneration` / `qwen4_exp_text`:

- 48 layers, hidden 2560, **512 experts / 10 active**, `moe_intermediate 640`
- 24 Q heads / 2 KV heads, head_dim 256, **262144** native context
- `layer_types`: 3 × `linear_attention` + 1 × `full_attention`, repeating (GDN hybrid)
- n-gram: 20 M × 2560 = **51.2 B** parameters, 128 shards, attached at layer 1
- **MTP: 1 layer**, "trained with multi-steps" — so k>1 reuses it autoregressively. Qwen publish no
  recommended k; the official vLLM recipe specifies **3**. We previously called **k=2 optimal** on
  decode evidence — **that is withdrawn**: on a fixed-work agent loop k=2 is the *worst*
  configuration measured here (48.6 ms/tok over 3 arms, against 43.6 for no speculation at all),
  while **n=5 is the best** at 31.9. Decode and per-turn latency rank the depths in opposite orders.
  See [which drafter for agent work](notes/which-drafter-for-agent-work.md).

Per-token byte budget (decode, `fp8head`): GDN 1.95 GiB, experts 1.24, hyper-connections 1.19,
QSA 0.59, `lm_head` 0.59, shared_expert 0.44. **The experts are 20% of it at c=1 and ~80% at
c=16** — each sequence pulls its own ten of 512 while the dense path amortizes.
