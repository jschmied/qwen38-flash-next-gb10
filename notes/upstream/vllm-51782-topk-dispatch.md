POSTED 2026-09-12 → vllm-project/vllm#51782 (comment). Revised after re-reading the thread: NNNtrance
posted a second follow-up at 07:53 showing exact `torch.topk` is within noise of the stock kernel,
which rules out the mechanism I was going to offer. Draft adjusted to say so rather than lead with it.

---

Two things from a single GB10 (sm_121, TP1). Different model here — Qwen3.8-Flash-Next — so this is
a reading of the dispatch, not a reproduction of your symptom.

**Q1: no, #55314 does not reach those paths.** `top_k_per_row_decode` and `top_k_per_row_prefill`
are defined in `csrc/libtorch_stable/sampler.cu` (lines 717 and 846, declared at `ops.h:466`), and
those are the only definitions in `csrc`. #55314 changes `persistent_topk.cuh`,
`cooperative_topk.cuh` and `topk_histogram_4096.cuh`; `sampler.cu` is untouched. It does carry the
same coarse-histogram / threshold-bin construct, with the same `smemFinalBinSize[0] <=
kNumFinalItems` guards — but I have not read its overflow branch, so I can only say the fix does not
reach that file, not that the defect is present in it.

**Why only long rows go there.** Leaving `persistent_topk` for `top_k_per_row_decode` needs three
conditions at once: the row exceeds `RADIX_THRESHOLD`, the cooperative launch would oversubscribe
the device, and `sharedMemPerBlockOptin < 128 KiB`. GB10 reports **100 KiB**, so it always satisfies
the third and only long rows satisfy the first two. Short and long rows on this hardware therefore
run *different kernels*, which is invisible from Python — worth knowing when you bisect by context
length.

That was going to be my explanation for your short-clean / long-corrupt split. **Your exact-`torch.topk`
result rules it out**, since replacing the selection at both call sites bypasses whichever kernel
would have run, and it moved nothing. So it supports your conclusion rather than competing with it:
the remaining variable is the size of the selected set, not which kernel selects it.

No view on your Q2 (attention-sink force-keep) — we have not tested it.
