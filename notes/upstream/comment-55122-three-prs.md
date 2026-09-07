DRAFT — user go given. GitHub vllm-project/vllm PR #55122 (2026-09-07).

There are now **three open PRs on this kernel** and none of them referenced the others, so linking
them here for whoever reviews any one of them.

| PR | approach | fixes the set | fixes the order | paths |
| --- | --- | --- | --- | --- |
| **#55122** (this) | remove the candidate buffers; rescan per key byte; index-ordered emission | yes | **yes** | persistent single- and multi-CTA, Filtered |
| **#55314** (@Dovis01) | keep the buffers; descend the key bytes until the bin fits; clip only on full-key equality | yes | no — output slots still come from `atomicAdd` | persistent, **cooperative_topk**, **histogram_4096** |
| **#53287** (@LopezCastroRoberto) | wider coarse histogram, exact fallback on overflow, buffered paths kept | yes | no | persistent |

The distinction that matters for #54521 is the middle column against the right one. All three produce
an exact top-k *set*. Only this PR also fixes the *order*, and the order is the half that forks greedy
decoding: the sparse attention sums the selected keys in output order, so two runs that select the
same set in a different sequence still diverge. #55314 keeps `atomicAdd(&shared_output_count, 1)` for
slot assignment; #53287 keeps the buffered paths.

I am not arguing the other two should close — **#55314 covers `cooperative_topk` and
`histogram_4096`, which this PR does not touch** (it bypasses the 2048-bin path rather than repairing
it, and `cooperative_topk` cannot even be exercised on sm_121: all 51 of its cases fail there with
`cooperative_topk launch failed: invalid argument`, a rejected cluster launch). If the set fix lands
from #55314 and the order fix from here, the union is better than either alone, and the two touch
mostly different functions.

What I would ask a reviewer to decide, since it is the only real conflict: whether the exactness fix
should keep the candidate buffers (#55314, #53287) or remove them (#55122). That choice determines
whether the order fix is a small addition or a rewrite, and it is a design call rather than a
correctness one.
