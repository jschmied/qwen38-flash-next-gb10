# Deterministic `persistent_topk` (kernel fix for vllm#54521 / finding 53–54)

vLLM's QSA block selection (`csrc/libtorch_stable/persistent_topk.cuh`, `topk_histogram_4096.cuh`)
hands out output slots with `atomicAdd` (thread-arrival order) and takes exact-key ties at the last
radix round first-come. On sm_121 that makes the selected ORDER vary run to run and, when more
elements share the threshold key than fit, the SET. The sparse attention sums the selected keys
in output order, so either forks the hidden state (bit-level proof: findings 52–54).

This directory is the kernel-side fix, as diffs against vLLM `main` (2026-09-02, c00091e0):

- `persistent_topk.det.diff` — decode / medium / large (multi-CTA) / filtered (>32k) paths
- `topk_histogram_4096.det.diff` — the ≤32k filtered path used by every prefill row

Two changes, applied at every emission site:
1. exact-key ties are collected and taken **lowest index first** (single-CTA paths: in-block
   bitonic sort of the tie buffer; multi-CTA path: per-CTA counts published before the barrier,
   slots from a prefix over CTAs, rank inside a CTA from a block scan);
2. every finished row is **sorted ascending** in shared memory before it leaves the kernel.

Cost: one in-block sort of ≤2048 ints per row (~66 compare-exchange steps at 1024 threads) plus
the tie sort when ties exist. Buffer overflow behaviour (>4096 candidates in a bin) is unchanged.

Standalone build (no vLLM rebuild): `build_det.py` compiles `topk_det.cu` + `bindings_det.cpp`
with `torch.utils.cpp_extension.load` into `_C_det` (op `torch.ops._C_det.persistent_topk`, same
signature as `_C.persistent_topk`). `torch_utils.h` is a minimal copy for torch 2.13's stable shim.
`test_det.py` checks: bit-identical across 6 calls, equal to an exact reference (value desc, index
asc), across rows {1, 8, 64} × lengths {1k … 40k} × k {512, 2048} × {random, tie-heavy}, and prints
whether the stock op reproduces itself on the same inputs.

Status: compiles for sm_121a; link + tests queued behind the measurement runs (`kdet` unit).
