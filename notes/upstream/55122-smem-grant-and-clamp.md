DRAFT — needs the user's go. GitHub review-comment reply, vllm-project/vllm PR #55122, in reply to
MaCoredroid's comment 4043382139 on csrc/libtorch_stable/persistent_topk.cuh:1693 (2026-09-18).

You are right on both counts, and the shared-memory one is a real bug. Fixed in `19588c89`.

**The granted size.** The port added `&max_seq_len, &smem_size` to the launcher's `args[]` and never
added the matching parameters to `FilteredTopKUnifiedKernel`, so `cudaLaunchKernel` silently ignored
both and the kernel kept using `FILTERED_TOPK_SMEM_DYNAMIC`. That constant is the *minimum* the
launcher requests — its own comment says so — and the launcher then clamps to the device with
`smem_size = min(want, cap)`. So on any device whose opt-in shared memory is under 128 KB the grant
is smaller than the constant and `det_select_row` caches past the end of what it was given. The two
dead `args[]` entries were the evidence that the change was half-landed; thank you for reading it.

It is latent rather than live, for the reason you give: the launcher does not select FilteredTopK on
GB10's 101,376 B limit, so neither your harness nor our runs reached it.

**The clamp.** Added, to `min(max_len, max_seq_len)` and floored at zero. To be precise about the
history: the pre-split wrapper did not clamp either — `filtered_topk_row` only ever ran its own
vectorised loop bound, so it could not overrun. The rescanning select indexes `length` directly, so
the bound has to be stated explicitly now. That is a difference the port introduced, not one it
removed.

**How far this is verified: a compile only.** The launcher and kernel instantiate and build clean for
`sm_121a`. That check does not discriminate — the previous code compiled too, since extra `void*`
entries in `args[]` are not type-checked — so it wants the in-tree test on a box that can select the
path. We cannot reach it here for the same 101,376 B reason.

Separately, thank you for re-running the harness against `85f61e24b`: 324 fallback plus 108
cooperative-control launches, and the 18 expected >64-CTA rejections.

AI assistance was used.
