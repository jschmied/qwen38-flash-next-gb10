POSTED 2026-09-24 (user go "do the post") as https://github.com/vllm-project/vllm/pull/55122#issuecomment-5809971218 — GitHub vllm-project/vllm PR #55122, perf numbers on the current head (2026-09-24). EDITED IN PLACE 2026-09-24 (user go "correct comment on 55122 in place"); body below is the live edited version.

*Edited 2026-09-24 after an audit of our notes: the paragraph on @k3dani's 21–28 % was wrong (struck through, corrected below), and two sentences are tightened. The measurements are unchanged.*

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

An eager-timed sweep over 26 shapes (random and tie-heavy data) gives PR / base from 0.67 to 1.00, so the PR is never slower (its decode cells sit on a ~10 µs launch floor, so the graph-replay table above is the measurement). Before
timing, the bench checked set equality with `torch.topk`, and that the PR's output is bitwise repeatable on tie
data.

**End to end.** Two interleaved server starts per arm, cold TTFT (prefix cache defeated by a salt):

| arm | TTFT 8k | TTFT 30k |
|---|---|---|
| stock | 2.727–2.746 s | 10.309–10.370 s |
| this PR | 2.732–2.736 s | 10.303–10.332 s |
| exact `torch.topk` + mask | 2.757–2.767 s | 10.652–10.704 s |

The PR is indistinguishable from stock in TTFT. The ~20 % kernel win holds at the 8k-context shapes, but it is ~4 % at the 8192-column shape that 30k exercises, and top-k is only about 0.3–0.5 % of 30k TTFT, so neither shows.
Against an exact `torch.topk`, the PR is 3–4 % faster at 30k and about 1 % faster at 8k.

~~This is smaller than @k3dani's 21–28 %. The likely reason is the workaround itself: theirs sorts each full row with~~
~~a stable descending sort ([`qsa_exact_topk.patch`](https://github.com/k3net/docai-evals/blob/master/experiments/2026-08-28-qwen38-flash-next-nvfp4-topk-nondeterminism-gb10/patch/qsa_exact_topk.patch)),~~
~~while `torch.topk` is the cheapest exact variant. So 3–4 % is the floor of the gap and 21–28 % applies to the~~
~~full-sort workaround. I did not run their variant.~~

**Correction:** their workaround is a full-row sort, but that does not explain the 21–28 % on its own. Their 09-12 arms also differed in image and vLLM version: the preview `0.1.dev20073` for the exact arm, the official `v0.29.0` for this kernel (per their eval-card). On a single image, in their 09-03 run above, an earlier revision of this PR beat the same workaround by 10–15 % (2253 vs 1952 tok/s at 6,082 tokens, 1932 vs 1756 at 24,416). So the sort explains part of the gap, and the rest cannot be attributed. I did not run their variant.

Not measured: decode, because the arms generate different text so per-token timings are not comparable; TP > 1;
and contexts beyond 30k.

Data and scripts: [finding](https://github.com/jschmied/qwen38-flash-next-gb10/blob/d7bbbb8103ef763f9aceb3fa7bc71ba33986a12a/notes/determinism-investigation.md#L6038) · [bench tools](https://github.com/jschmied/qwen38-flash-next-gb10/tree/d2b6a53acf597a95571ac9b89a9988ff198781c9/tools/topk55122)

*AI assistance was used in preparing this comment; the measurements were run and checked by me.*

