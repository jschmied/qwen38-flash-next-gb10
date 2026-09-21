DRAFT — needs the user's go. vllm-project/vllm PR #55430 (2026-09-21). Closing.

Closing this. The re-measurement you asked for is done and the design cannot clear the bar, for a
reason that is arithmetic rather than tuning.

I profiled one 28,933-token prefill on GB10 (`sm_121`, TP1, union off, MTP off) and grouped GPU
kernel time — 9.841 s over 24,473 launches:

| group | % of kernel time |
|---|---:|
| MoE grouped GEMM + routing | 36.4 |
| hyper-connections | 13.7 |
| **QSA attention + index** | **12.2** |
| dense GEMM | 11.2 |
| blockwise FP8 GEMM + quant | 11.1 |
| GDN / linear attention | 8.3 |

The tile-union replaces the QSA attention kernel only, so 12.2 % is its entire budget. At the
measured 1.45× kernel ratio the end-to-end ceiling is **3.78 %**; at 2× it is 6.1 %; an infinitely
fast kernel would still buy only 12.2 %. Against a ">3 %" bar the design is marginal by construction.

End to end, after rebasing the patch for #55272 (which had broken the union path outright, so the
earlier 1.45×-derived figures described code that could not run), with byte-identical 29,030-token
prompts per arm and the union path confirmed active in the log:

| arm | median TTFT |
|---|---:|
| union off | 10.149 s |
| union on | 9.990 s |

**−1.6 %**, consistent with the −1.7 % at 30k measured earlier, i.e. about 42 % of the ceiling above;
the remainder goes to the sort/pack/build glue and ~3.6 ms of added idle per request. Earlier cells:
8k −1.0 %, 16k +0.5 %.

So your read on 2026-09-05 — too small for the complexity — was right, and the numbers now say why
rather than just how much. The RFC (#55394) I will leave open only if the tile-union idea is of
interest for a part where the sparse-attention kernel is a larger share of prefill; on this one it
is not.

Thanks for the review time.

_AI assistance (Claude Code) was used in preparing this comment; the measurements are from a GB10._
