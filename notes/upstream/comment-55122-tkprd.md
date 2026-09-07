DRAFT — needs the user's go. vLLM PR #55122 reply to @gau-nernst (2026-09-07).

Thanks — I checked it. **`top_k_per_row_decode` is the faster kernel, but it is not deterministic**,
so it fails the first half of the requirement. I drove it through the call convention this model's
own AMD path already uses (`vllm/models/qwen4_exp/amd/ops/qsa.py`, where it is the `else` branch to
`persistent_topk` on identical inputs), on a GB10 (sm_121, TP=1).

**Determinism: 0 of 56 shapes.** Six identical calls never reproduce, across rows {1, 8, 64} ×
n {1k … 40k} × k {512, 2048} × {random, tie-heavy}, plus all-equal. Not an uninitialised-output
artefact: re-run with the output pre-filled with two different sentinels, 0 unwritten slots either
way and the sets agree between sentinels — 9,824 differing slots over five pairs at
rows=1 / n=8192 / k=2048, 634,059 at rows=64.

**And it is the same defect class.** The selected *set* equals the exact reference on random inputs
(the order does not), but **differs from it on every tie-heavy shape** — which is #51782 and the
kernel half of #54521, i.e. the bug this PR exists to fix.

**Speed, median µs of 5 × 50 launches:**

| | vs our deterministic kernel | vs stock `persistent_topk` |
| --- | --- | --- |
| n ≤ 8k, any rows | 0.15–0.45× | 0.40–0.80× |
| 16k–32k, 1–8 rows | 0.48–0.92× | **1.10–1.48× (slower)** |
| 16k–32k, 64 rows | 0.28–0.54× | 0.62–1.00× |

So it beats us everywhere, and beats *stock* only below ~16k or at high row counts.

**On the 3×.** That figure is from this PR's own cost table and it is a per-call microbenchmark. The
end-to-end measurement is [further up the thread](https://github.com/vllm-project/vllm/pull/55122#issuecomment-5536961535):
server-level A/B, three starts per arm, **no TTFT change and no per-turn cost** — 8k TTFT 3.10 / 3.29 / 3.10 s
stock against 3.12 / 3.35 / 3.12 s det, 30k 11.24 / 11.55 / 11.26 against 11.22 / 11.62 / 11.27, and
seconds per agent turn 2.70 / 2.68 / 2.68 against 2.69 / 2.54 / 2.66. Twelve QSA layers per step put
this kernel well under a percent of the step, so the per-call ratio does not reach the clock. If a
3× per-call cost is unacceptable in principle even at zero end-to-end cost, say so and I will
withdraw the PR in favour of the direction below.

**The direction your suggestion actually opens.** The interesting move is the reverse one: give
`top_k_per_row_decode` the same treatment — index-ordered emission from an exact pivot — since it is
already the non-CUDA branch for this model and is the faster kernel. That would be deterministic
*and* faster than what I propose here. I am happy to do it as a follow-up PR; it is a different
kernel and a different review, so I would rather not fold it into this one. I have not looked at the
Minimax MSA bitonic top-k yet — pointer welcome if you think it is the better base.

*AI assistance was used for this work; every line was reviewed by me.*
