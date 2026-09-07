DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122, reply to gau-nernst (2026-09-07).

Good question, and I had not measured either — so I did. GB10 / sm_121, TP1, `_C_det.so` built from
this branch's head, 5 × 50 launches, median µs. Harness and full output:
[`alt_cmp.py`](LINK_CMP) / [`alt.txt`](LINK_ALT), config sweep [`alt2.txt`](LINK_ALT2).

**Both are deterministic. Both are much slower here.**

| rows × n × k | this PR | `torch.topk` | MSA bitonic | stock |
| --- | --- | --- | --- | --- |
| 1 × 8,192 × 2048 | **10.5** | 96.6 (9.2×) | 191 (18.2×) | 10.3 |
| 8 × 8,192 × 2048 | **12.4** | 94.2 (7.6×) | 194 (15.7×) | 12.3 |
| 8 × 32,768 × 2048 | **22.7** | 144 (6.4×) | 735 (32.4×) | 20.7 |
| 64 × 8,192 × 2048 | **20.6** | 170 (8.3×) | 357 (17.4×) | 18.6 |
| 64 × 32,768 × 2048 | **77.9** | 363 (4.7×) | 1413 (18.1×) | 57.3 |

Across the 27-shape grid `torch.topk` is **4.4–9.9×** and the bitonic top-k **5.0–35.6×**. Those
figures include what the op's contract needs and `torch.topk` does not do — the ragged-length mask
and the ascending-index sort. Stripped of both, bare `torch.topk` on a dense row is still
**1.9–4.7×**, so the gap is not the wrapper.

**`torch.topk`** passed every case: bit-identical over 6 calls, exact set, and after the sort it
matched the index-canonical reference on all 58 shapes including the tie-heavy and all-equal ones.
Worth saying plainly — it *would* fix the determinism bug. I would not rely on the tie behaviour as
a contract (it is not documented and I tested one PyTorch build on one device), but on this box it
holds. It is the cost that rules it out: at the decode shape this op runs at, ~90 µs against ~12 µs,
on the critical path of every layer.

**The MSA bitonic top-k** I ported out of `minimax_m3/common/ops/index_topk.py` — the same
`_bitonic_merge` primitives and the same streaming structure (sort a `BLOCK_SIZE_K` tile, merge
against the running winners, keep the top half), with the paged/causal/init-local plumbing removed
so it selects from a flat ragged row. Three things came out of it:

- It is **deterministic and exact by value**. It shows 29/58 "set mismatches" against my reference
  only because the bitonic network breaks ties by network position, not by lowest index — I checked
  the selected values against the exact top-k multiset and they match everywhere, with no duplicates
  and no out-of-range indices. So it is a *valid* top-k, just not the index-canonical one. For this
  bug that is enough; reproducibility is the requirement, not a particular tie choice.
- The cost is not my config. Sweeping `BLOCK_SIZE_K` ∈ {2T, 4T} × `num_warps` ∈ {2, 4, 8, 16}, the
  best cell is still **8.9–31.4×** this PR.
- As shipped it cannot serve `k = 2048`: `BLOCK_SIZE_T = next_pow2(topk)` under
  `static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)` needs `BLOCK_SIZE_K ≥ 4096`, and M3's autotune
  configs stop at 2048. QSA runs k = 2048. Forcing `BLOCK_SIZE_K = 4096` by hand does work — it is
  the 13.6× and 31.4× cells above — but each config costs minutes of Triton compile time.

The reason the gap is that large is that both alternatives materialise and order data this op never
needs ordered. This PR's single-CTA path rescans the row per key byte and writes straight into final
positions — no candidate buffer, no sort. That is also why it ended up in the same range as the stock
kernel rather than above it.

Happy to run either on other shapes if there is one you think would flip it.
