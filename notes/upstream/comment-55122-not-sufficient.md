DRAFT — needs the user's go. vllm-project/vllm PR #55122.

---

A correction to this PR's own claims, before anyone merges it on the strength of them.

**This kernel alone does not make greedy decoding reproducible on GB10.** I have been measuring
end-to-end reproducibility with per-position `prompt_logprobs` over 8 identical sequential requests
(2.5k-token prompt, prefix caching on, MTP=3), counting positions where the modal top-1 token
disagrees between repeats:

| build | disagreeing positions |
| --- | --- |
| stock, no determinism patches at all | **333** (first divergence at position 2, spread 8.79) |
| **this PR's kernel alone** | **330** (first divergence at position 4, spread 8.54) |
| all four patches we run | **0** |

So on this stack the deterministic top-k is *one necessary component of a set*, not the fix. I had
presented it as the latter, and that was wrong.

Two concrete consequences for this PR:

**1. `Fixes #54521` should not stay in the description.** On merge it would auto-close an issue this
PR does not close. I will change it to a reference unless a maintainer prefers otherwise.

**2. The "three independent defects" sentence is wrong**, and wrong in a way that matters. On our
stack it takes four: this kernel, the FlashInfer CUTLASS MoE fused finalize (#54945 / PR #54948),
the FlashInfer autotune cache key (which must include `use_fused_finalize` — without it, flipping
that flag dies at init with `Invalid gemm2 profile id`), and **a PLE offload semaphore reset**. That
last one is in **no released vLLM and not on main**: `vllm/v1/ple_offload/` does not exist upstream
(zero files in the current nightly wheel, 404 on `main`), #53899 is open with conflicts, and the
semaphore fix sits on top of it as a PR against that fork branch. Anyone reproducing our
reproducibility result today cannot get there from vLLM alone, and the PR text implied they could.

**What this PR still claims, and I stand behind:** the kernel is deterministic call-to-call (81/81
shapes) and index-canonical where stock is neither; the selected set matches the exact reference
16/16 including the tie-heavy shapes from #51782; and after the two commits added today it is
0.72–1.78× stock across 43 cells, at or below stock on 27 of them. That is a kernel correctness fix
with a measured cost, and it is worth having on its own terms.

@LopezCastroRoberto — this is also, belatedly, evidence for the point you made: changing the default
on a reproducibility argument was the wrong framing, and the opt-in shape you proposed does not
depend on the claim I got wrong.

<!-- AI disclosure: produced with AI assistance; I reviewed every line and ran every number quoted. -->
