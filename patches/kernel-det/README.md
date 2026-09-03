# Deterministic `persistent_topk` (kernel fix for vllm#54521 / finding 53–54)

vLLM's QSA block selection (`csrc/libtorch_stable/persistent_topk.cuh`, `topk_histogram_4096.cuh`)
hands out output slots with `atomicAdd` (thread-arrival order) and takes exact-key ties at the last
radix round first-come. On sm_121 that makes the selected ORDER vary run to run and, when more
elements share the threshold key than fit, the SET. The sparse attention sums the selected keys
in output order, so either forks the hidden state (bit-level proof: findings 52–54).

This directory is the kernel-side fix, as a diff against vLLM `main` (2026-09-02, c00091e0):
`persistent_topk.det.diff` (v2 — after review; v1 still admitted tie candidates into fixed
buffers in arrival order before ranking them, so >buffer exact ties stayed scheduler-dependent).

- **Every single-CTA row** (persistent decode/medium paths, and the filtered kernel for
  float) goes through `det_select_row`: a radix select that **rescans the row for each of the
  four key bytes** (no candidate buffers → no truncation, exact pivot), then one index-ordered
  block scan that emits all elements above the pivot and the lowest-index `fin` elements equal
  to it, then sorts the row ascending. Five reads of the row + one in-block sort of ≤2048 ints.
- **Multi-CTA rows** (> RADIX_THRESHOLD) keep the existing radix rounds (histograms over the
  chunk in shared memory, no truncation) and get a deterministic emission: per-CTA `>`/`==`
  counts published before the barrier, slots from a prefix over CTAs, `==` elements ranked by
  index with a block scan, CTA 0 sorts the finished row.

`topk_histogram_4096.cuh` is unchanged (its float instantiation is no longer reached).

Standalone build (no vLLM rebuild): `build_det.py` compiles `topk_det.cu` + `bindings_det.cpp`
with `torch.utils.cpp_extension.load` into `_C_det` (op `torch.ops._C_det.persistent_topk`, same
signature as `_C.persistent_topk`). `torch_utils.h` is a minimal copy for torch 2.13's stable shim.
`test_det.py` checks: bit-identical across 6 calls, equal to an exact reference (value desc, index
asc), across rows {1, 8, 64} × lengths {1k … 40k} × k {512, 2048} × {random, tie-heavy}, and prints
whether the stock op reproduces itself on the same inputs.

`test_det.py` adds the review's adversarial cases: all-equal rows must return exactly `[0..k)`
100× on every path, and pivot-tie populations of k−1, k, k+1, 2048, 2049, 3708, 3709, 4096,
16384, 16385 around every buffer size the original kernels used.

Status (2026-09-03): v2.3 builds and links against torch 2.13 / CUDA 13 on the box; `test_det.py`
177 / 177 (`test_results.txt`); `bench_det.py` in `bench_results.txt` — det costs 1.3–4× the stock
call (8→10 µs at n=1k, 18.5→72 µs at n=32k/k=2048, single row), the multi-CTA path (> 32k) is the
cheaper one. Model-level estimate ≈ +1.5 % decode at 32k ctx, ≈ +1.8 % TTFT at 7.5k (finding 63).
Build notes: the standalone glue needs `-DUSE_CUDA` (the CUDA stream getter in torch's stable shim
is guarded by it; the generic stream getter returns an opaque handle and segfaults in the memset)
and the launcher caps the dynamic-smem request at `sharedMemPerBlockOptin − static __shared__`.
On GB10 the opt-in is 101,376 bytes, so the `num_rows > 32` filtered path is never dispatched here.

Follow-up (not in this diff, on purpose): the filtered kernel keeps its `VEC_SIZE` /
`UsePredicatedShortLoads` instantiation ladder, `FilteredTopKTraits`, `vec_t` and
`ComputeFilteredTopKVecSize`, which the deterministic path no longer uses. Removing them deletes
~200 lines and the launcher's instantiation dispatch; it is a separate, mechanical commit so the
correctness diff stays reviewable. Only cost of leaving it: a few identical kernel instantiations
(compile time).
