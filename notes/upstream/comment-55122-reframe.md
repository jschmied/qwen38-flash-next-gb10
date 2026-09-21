DRAFT — needs the user's go. vllm-project/vllm PR #55122 (2026-09-21).

Retitled and restated the purpose of this PR. The determinism framing it opened with overstates the
case on the evidence since, so rather than leave the body arguing that, here is what changed.

Three independent readings, none of them ours alone:

1. We censused the top-k boundary on real traffic: **0 ties at the k-th value across 6,192 selecting
   rows** at 16k context; 93 % of rows had fewer visible blocks than k, so selection was a no-op.
2. @xueyangcs, who reported #51782, states the HPC-Ops TopK contract is **set-exact only** — with
   ties any valid top-k set is permitted, and set-stability was never guaranteed.
3. @NNNtrance replaced the selection with an exact `torch.topk` on GLM-5.3-Flash across three DGX
   Sparks: **9/6 bad turns vs 11/7 stock, within noise**, while `index_topk 8192` gives 2/3.
   Selection accuracy was not the cause of their symptom.

So the tie-ordering defect is real at the kernel level but does not look reachable on the traffic
that motivated it. What survives is the performance argument, and that part is independently
measured rather than ours: @k3dani found this kernel **21–28 % faster than the exact-topk
workaround** on a second GB10, and the end-to-end server A/B shows no TTFT or per-turn cost.

**Please judge it as a performance PR with a correctness side-benefit.** It does not close #54521,
as the bottom of the body already says. If the complexity is not worth that trade, closing is the
right outcome and I will not argue it.

@k3dani, @MaCoredroid — flagging since your measurements and review are what the remaining case rests
on; the change is to the framing, not to the kernel.

_AI assistance (Claude Code) was used in preparing this comment._
