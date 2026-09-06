# POSTED 2026-09-06 — PR #55180 server-level evidence (https://github.com/vllm-project/vllm/pull/55180#issuecomment-5557288939)

Server-level numbers for the swizzle, since the review asked what it does end to end rather than in a microbench.

Setup: Qwen3.8-Flash-Next (FP8-mixed checkpoint), one GB10, vLLM main (dev401) with this PR's kernel loaded as an
overlay op, no speculation, prefix cache off, two server starts per arm, three requests per cell.

| chunk (`max_num_batched_tokens`) | kernel | TTFT 7.5k prompt | TTFT 29k prompt |
| --- | --- | --- | --- |
| 4096 | stock | 2.62–2.66 s | 10.3–10.4 s |
| 4096 | this PR | 2.63–2.65 s | 10.3–10.4 s (gate closed: 4k × 2560 = 10 MiB of A, below the 12 MiB threshold) |
| 16384 | stock | 2.47–2.59 s | 10.81–10.84 s |
| 16384 | this PR | 2.46–2.76 s | **9.50–9.52 s (−12 %, 2 starts; a third, profiled start: 9.83 vs 13.08 s stock under the profiler)** |

At 16k chunks the gate opens for the projections above the 24 MiB L2 (`in_proj_qkv` 25 MiB ×36, `q_proj` 30 MiB
×12) and the 29k prompt drops 1.3 s; a concurrent 29k+7.5k pair drops 13.3 → 12.0 s. The 7.5k prompt is a single
chunk where the stock kernel is only mildly degraded, so the difference stays inside start-to-start noise. Net effect
for this model: the 16k chunk becomes strictly better than 4k at every prompt size (before, 4k won at 29k because the
stock kernel collapsed on the large-M chunk). Bit-identical outputs, as in the standalone check.

Kernel-level, from a torch-profiler trace of one 29k prefill per arm (16k chunks): the blockwise-FP8 GEMM kernel
totals 2,532 ms stock vs 1,011 ms with this PR over the request (192 calls, 13.2 → 5.3 ms per call); the largest shape,
[16384, 2560] × [2560, 16384] (the fused GDN qkv+z projection, 40 MiB of weight), goes 27.9 → 8.8 ms per call. Every
other kernel family is unchanged within noise.

One caveat for anyone reproducing with an overlay rather than the built kernel: vLLM's compile cache does not key on
an env-gated Python branch, so A/B arms need separate `VLLM_CACHE_ROOT`s or the second arm silently reuses the first
arm's graph.
