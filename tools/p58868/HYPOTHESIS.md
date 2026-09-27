LD58868 (vllm#58868: fault mmap'd weight pages on the CPU before the H2D copy on integrated GPUs), 2026-09-27, before
the run. User: "post, then 1 and 2" (2 = queue a load-time A/B). Clone venv rssm (1ea7c63f4 + our overlay); the PR diff
applies with offsets to our overlaid linear.py / vocab_parallel_embedding.py / weight_utils.py / routed_experts.py /
parameter.py; utils.py gains copy_weight_. Arms stock vs pr58868, alternating, 2 starts, every start from a cold page
cache (launcher drops caches). Metric: "Loading weights took X s" (main model; the second line is the MTP drafter).
Baseline from tonight's logs: 516 s main + 62 s drafter.
H: main weight load -30 ... -55 % (PR/gitbisector: ~2x at TP1 on a Spark); drafter load similar ratio; greedy sanity hash
identical between arms (the weights are the same bytes). Out of range: no change (then the mapped PLE path or our loader
patches bypass the copy sites) or a hash difference (then a copy site changed semantics).
