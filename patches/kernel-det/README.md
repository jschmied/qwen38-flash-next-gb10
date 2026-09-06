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

**Upstream: PR https://github.com/vllm-project/vllm/pull/55122 (2026-09-03).** `test_upstream_top_k_per_row.py` is the PR's test file; `detplugin.py` runs its new cases against the standalone `_C_det`.

Status (2026-09-06): **v2.4** — host-guard fix. v2.3 rejected every call whose row is shorter than
`TopK` (`persistent_topk_det: chunk_size 256 smaller than TopK 512`): the main build's block-level
QSA indexer calls `TopK = token_topk / compress_ratio = 512` with 256-block rows at warm-up, so the
server never started. The `chunk_size >= TopK` requirement only exists on the cooperative large path
(`max_seq_len > RADIX_THRESHOLD`, where CTA 0 sorts the candidates in its chunk buffer); rows at or
below the threshold take the single-CTA `det_select_row` / trivial `seq_len <= TopK` case and never
touch that buffer. The guard is now conditional on the path. `test_det.py` gained 33 short-row cases
(rows ≤ k, all three TopK); the old matrix skipped `k >= cols` and so never saw the shape. 210/210
pass (`test_results_v24.txt`), built in ~1 min against torch 2.13 / CUDA 13 on the box. **On the PR since
2026-09-06** as `c564e5c1`, with `test_persistent_topk_short_rows` (33 cases: rows 256..2048 at TopK 512/1024/2048);
the same push finally carried the 09-03 review commit `afd92810`, which had never reached the PR.

Status (2026-09-03): v2.3 builds and links against torch 2.13 / CUDA 13 on the box; `test_det.py`
177 / 177 (`test_results.txt`); `bench_det.py` in `bench_results.txt` — det costs 1.3–4× the stock
call (8→10 µs at n=1k, 18.5→72 µs at n=32k/k=2048, single row), the multi-CTA path (> 32k) is the
cheaper one. Model-level estimate ≈ +1.5 % decode at 32k ctx, ≈ +1.8 % TTFT at 7.5k (finding 63).
Build notes: the standalone glue needs `-DUSE_CUDA` (the CUDA stream getter in torch's stable shim
is guarded by it; the generic stream getter returns an opaque handle and segfaults in the memset)
and the launcher caps the dynamic-smem request at `sharedMemPerBlockOptin − static __shared__`.
On GB10 the opt-in is 101,376 bytes, so the `num_rows > 32` filtered path is never dispatched here.
`RADIX_THRESHOLD` is lowered 32768 → 16384 in the diff: the deterministic multi-CTA path is cheaper
than the single-CTA `det_select_row` above 16k (`bench_results_threshold.txt`; 8192 was worse for
64-row batches). Wiring for vLLM: `tools/determinism/qsadet_patch.py` + `VLLM_QSA_DET_TOPK=1`.

Follow-up (not in this diff, on purpose): the filtered kernel keeps its `VEC_SIZE` /
`UsePredicatedShortLoads` instantiation ladder, `FilteredTopKTraits`, `vec_t` and
`ComputeFilteredTopKVecSize`, which the deterministic path no longer uses. Removing them deletes
~200 lines and the launcher's instantiation dispatch; it is a separate, mechanical commit so the
correctness diff stays reviewable. Only cost of leaving it: a few identical kernel instantiations
(compile time).


## For validators on another box (e.g. the `vllm/vllm-openai:qwen38-flash-next` image)

1. `python build_det.py` in this directory with the image's Python (needs `nvcc` for sm_121a; ~5 min;
   set `DET_ARCH` for another GPU). It builds `build/_C_det.so` against the installed torch.
2. `python test_det.py build/_C_det.so` — expect 0 FAILS, and every `stock identical x3=False`.
3. Wire it in: `python ../../tools/determinism/qsadet_patch.py` (finds `qsa.py` in the running vLLM;
   `VLLM_QSA_PY=` overrides), then serve with `VLLM_QSA_DET_TOPK=1 VLLM_QSA_DET_LIB=<path>/_C_det.so`.
   The log prints `QSADET active: <lib>` once. `qsadet_patch.py off` removes it.
