DRAFT — needs the user's go. Reply on #54521 to davidcanar (2026-09-04 18:46).

Thanks — a third platform without QSA is the strongest evidence yet that the issue title is too narrow, and it matches our GB10 data. Two things that may help you cut it down:

**1. Sequential vs concurrent divergence are different mechanisms; separate them first.** On GB10 (sm_121, TP=1) the same symptom had three kernel causes — a MoE finalize that reduced in a non-deterministic order (#54945/#54948), the GDN align-block seeding (#54076/#53798), and `persistent_topk` (#55122). With those fixed, *strictly sequential* greedy on an idle server reproduces bit-for-bit across restarts (six of six), while *concurrent* greedy still diverges. Your `minrepro.py` sends the five requests sequentially, so if you get 5/5 distinct on an otherwise idle server, an order-nondeterministic kernel (atomics or a split reduction whose partial order is not fixed) is in your path, and the batch-shape explanation below is not it. On your stack the candidates by that criterion are the Triton ragged sparse-MLA decode patch (any split-K / atomic accumulate in it?), the MoE combine after the experts (your `fused_experts_op` check covers this only if that op includes the top-k weighted reduce), and the collective (RCCL can pick a different algorithm per call size). Run the five reps with `--max-num-seqs 1` and nothing else on the server; then with one concurrent dummy stream. Which of the two flips tells you the channel.

**2. The batch-shape channel is real and measurable in isolation.** A decode row at M=1 and the same row inside an M=8 verification block (or a chunk boundary that moved because another request was scheduled) need not go through the same arithmetic. Measured on sm_121 with the same first row through M = 1…4096 on the model's dense shapes (`tools/gemm_m_invariance.py` in https://github.com/jschmied/qwen38-flash-next-gb10 — only needs a vLLM venv with `_C`):

| path | row 0 identical to the M=1 result | max \|diff\| (bf16 out) |
| --- | --- | --- |
| per-channel FP8 (CUTLASS sm120) | every M | 0 |
| blockwise FP8 128×128 (CUTLASS sm120) | M=1 only, differs from M=2 on | 1.0–1.5e-2 |
| BF16 cuBLAS | M=1 only, differs from M=2 on | 0.25–2.0 (unscaled randn) |

That is deterministic per shape — it cannot produce five distinct outputs from five identical sequential requests — but it does produce them the moment the scheduler batches differently between reps, and with MTP it is the E == V ≠ A pattern reported in #54928. Your W4A16 kernel on gfx1151 would be one more row in that table; the script is trivially adaptable.

Happy to run any cell on the GB10 in return.
