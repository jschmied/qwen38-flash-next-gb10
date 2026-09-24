DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122, perf numbers on the current head (2026-09-24).

Perf numbers for the current head (`b2312b2de`), which nobody had measured since the port. This is on one DGX Spark
(GB10, sm_121, TP=1), with vLLM main `1ea7c63f4` and Qwen3.8-Flash-Next.

**Kernel, GPU time.** Timed by CUDA-graph replay, so host launch overhead is out. The PR head and its merge base
`3df4ae15` were built standalone with identical flags. The profiler shows the same `persistent_topk_kernel<512,4u>`
in both, so no routing differs.

| shape (rows × columns) | merge base | this PR | PR / base |
|---|---:|---:|---:|
| decode 4 × 2048 | 6.59 µs | 4.89 µs | 0.74 |
| decode 64 × 8192 | 15.03 µs | 11.84 µs | 0.79 |
| prefill 4096 × 2048 | 366.6 µs | 294.6 µs | 0.80 |
| prefill 4096 × 8192 | 633.9 µs | 610.6 µs | 0.96 |

An eager-timed sweep over 26 shapes (random and tie-heavy data) gives PR / base from 0.67 to 1.00, so the PR is never slower. Before
timing, the bench checked set equality with `torch.topk`, and that the PR's output is bitwise repeatable on tie
data.

**End to end.** Two interleaved server starts per arm, cold TTFT (prefix cache defeated by a salt):

| arm | TTFT 8k | TTFT 30k |
|---|---|---|
| stock | 2.727–2.746 s | 10.309–10.370 s |
| this PR | 2.732–2.736 s | 10.303–10.332 s |
| exact `torch.topk` + mask | 2.757–2.767 s | 10.652–10.704 s |

The PR is indistinguishable from stock end to end; the kernel win is too small a share of the step to show.
Against an exact `torch.topk`, the PR is 3–4 % faster at 30k and about 1 % faster at 8k.

This is smaller than @k3dani's 21–28 %. The likely reason is the workaround itself: theirs sorts each full row with
a stable descending sort ([`qsa_exact_topk.patch`](https://github.com/k3net/docai-evals/blob/master/experiments/2026-08-28-qwen38-flash-next-nvfp4-topk-nondeterminism-gb10/patch/qsa_exact_topk.patch)),
while `torch.topk` is the cheapest exact variant. So 3–4 % is the floor of the gap and 21–28 % applies to the
full-sort workaround. I did not run their variant.

Not measured: decode, because the arms generate different text so per-token timings are not comparable; TP > 1;
and contexts beyond 30k.

Data and scripts: [finding](SHA_LINK_FINDING) · [bench tools](SHA_LINK_TOOLS)

*AI assistance was used in preparing this comment; the measurements were run and checked by me.*
