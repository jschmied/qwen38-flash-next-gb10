# What speculation is worth on Flash-Next, and what limits it

## Measured (fp8head, one box, one method, 2026-08-28)

| arm | c=1 decode t/s | c=1 TTFT | c=16 aggregate | c=16 TTFT |
| --- | --- | --- | --- | --- |
| MTP off | 26.4 | 1.87 s | 96.6 | 9.71 s |
| MTP k=2 | 38.0 | 1.82 s | 99.1 | 6.79 s |
| gain | **+44%** | — | **+2.6%** | −30% |

input-len 4000, output-len 512, `--max-num-seqs 16`, temp 0.6, 6 requests at c=1 / 32 at c=16.

**Speculation does not hurt at saturation — it stops paying.** The reason is structural: the
experts are 20% of per-token bytes at c=1 but ~80% at c=16 (see `moe-backend-axis.md`), and by
then the batch already commits the machine. Draft tokens need idle capacity to convert into
throughput, and at c=16 there is none.

It still cuts c=16 TTFT by 30%, which is the reason to keep it on for agent work whatever it does
for aggregate throughput — agent turns emit ~130 tokens, so they live in the TTFT-bound regime.

## The limit we are running into

```
[speculator.py:117] Fused multi-step draft decode is not supported by attention backend(s)
QWEN38_FLASH_NEXT_EXP_QSA_STATE; falling back to rebuilding attention metadata between draft steps.
```

Every draft step rebuilds attention metadata instead of using the fused multi-step path. The cost
is per *draft step*, so it grows with `k` — which is the mechanism that would pin the optimum at a
low `k`. Sweeping `k` to test this.

QSA carries a stateful decode backend (`..._QSA_STATE`), and the fused path has no implementation
for it. This is a genuine gap for this model family, not a misconfiguration on our side.

## Seven kernels JIT-compile during inference

`enable_jit_warmup=True` is set in the kernel config, yet vLLM's own `jit_monitor` reports 20
warnings across seven kernels — including all three QSA decode kernels:

```
_qsa_sparse_paged_gqa_splitk_kernel   _qsa_mqa_paged_kernel   _qsa_merge_splitk_kernel
_expand_qsa_indices_kernel   _causal_conv1d_update_kernel   _fused_post_conv_kernel
_topk_topp_kernel
```

The warmup does not cover the Flash-Next-specific kernels. First requests after a start pay JIT
cost; discard them before measuring.

## Not our bug: the SM121 QSA decode issue

MiaAI-Lab (`Qwen3.8-Flash-Next-Dual-DGX-Sparks`, 2026-08-28) routed GB10 **off** FlashInfer
TRT-LLM sparse decode onto a packed one-query Triton kernel (sglang#36845), fixing a token-0
correctness bug on SM121.

We are not exposed. Every `TRTLLM` occurrence in our log is in a *potential backends* list and is
never selected; our QSA decode already runs vLLM's own Triton kernels. **Their fix moves SGLang
onto the path we are already on.**
