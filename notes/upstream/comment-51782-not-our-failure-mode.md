DRAFT — user go (same). vllm#51782 (2026-09-10).

@xueyangcs Thank you — and the answer lets us close the question in your favour: **set-exactness is
enough for our case, and adding set-stability would not have fixed us.** We measured where our
nondeterminism actually comes from, and it is not tie-breaking.

**1. The top-k boundary does not tie on our traffic.** We instrumented the QSA indexer's `_topk` and
censused the boundary over 400 calls (GB10 / sm_121, TP 1, MTP 3, k = 512 blocks, two ~13.6k-char
prompts): **87,257 rows, of which 6,192 performed a real selection** (the rest had fewer visible blocks
than the budget, so everything was taken). Rows with *any* value equal to the k-th: **0**. Not "tied but
resolved consistently" — never tied. Exact ties between float32 logits out of a real GEMM are simply
rare.

**2. The scores differ run to run; the selection never differs given identical scores.** Hashing the
*input scores* and the *selected index set* per call across 7 byte-identical greedy requests: all **13**
comparable prefill calls have **different input scores**, and **zero** have identical scores with a
different selection. With our determinism fixes enabled, all 13 are bit-identical on both.

So the divergence originates **upstream** of the selection — reduction-order rounding in producing the
scores — and the top-k is only where it becomes visible. A set-stability guarantee would make the kernel
reproducible **given identical inputs**, which is exactly the condition we do not have.

**The practical consequence for your design decision:** you noted set-stability and order-determinism
would cost comparisons and latency. On this workload they would buy us nothing, because ties do not
occur and the inputs are not identical. We would rather have the latency. If someone reports a case with
genuinely tie-heavy logits — low-entropy or heavily quantised scores — that is where the guarantee would
earn its cost, and it is worth asking them for a tie census before paying for it.

Bounds: 16 k context, two prompts, ~6 × 10³ selecting rows, one model on one GPU family. A tie rate
indistinguishable from zero at that sample size is not zero at 10⁶.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
