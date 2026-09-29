# Marlin MoE determinism (2026-09-29, before the run)

Trigger: the `marlin` A/B (§5ah) gave different greedy text in two starts of the same config; CUTLASS reproduced.
Atomic add is excluded by reading (`use_atomic_add=False` hard-coded in both Marlin MoE GEMMs). `det_marlin_moe.py`:
Flash-Next expert shapes, random NVFP4 weights, fixed inputs; M = 1, 4, 55, 512; run twice as two processes.

- H-kernel: `repeat_classes` > 1 at some M, or the two processes' `xproc` hashes differ → the Marlin MoE GEMMs themselves
  are nondeterministic (find which of gemm1 / gemm2 / moe_sum next).
- H-order: repeats equal, but `permute_equal` false while `align_orders` > 1 → per-token results depend on where the
  token lands in moe_align's (atomic, unordered) per-expert list, e.g. via block_size_m tiling or split-K per tile.
- H-outside: everything equal in and across processes → the Marlin kernels are deterministic; the server divergence
  comes from the serving context (CUDA graphs + shared workspace/locks, a different MoE path at prefill, the router
  running on padded batches). Expected: H-outside or H-order, the kernel's split-K reduce uses a fixed-order lock.

## `marlinrep` (2026-09-29, before the run): within one start, or only across starts?

Result above: H-outside (the kernel is deterministic). Probe `repro.py`: greedy A, A, B, A with a unique `cache_salt`
each (cold path, same tokens), then A twice unsalted (cache hit). Arms: Marlin + MTP K5 (as `marlin`), Marlin without
speculation, CUTLASS + K5 (control; must give one A class per start and the same A0 across starts); 2 starts each.
- H-start: A is one class within each Marlin start, but A0 differs between the two starts → something fixed at
  startup (weight repack, workspace layout, graph capture) differs per start.
- H-run: A has > 1 class within a Marlin start → a runtime race (shared workspace/locks under CUDA graphs, stream
  overlap); if only with MTP, the drafter's own Marlin MoE call is involved.
Expectation: H-start, a guess: nothing measured so far separates the two.

`marlinrep` was stopped after start 0's Marlin arm (A one class within the start, A0 62837025…): its no-speculation arm
cannot start with `--use-replayssm` (RecoverSSM requires speculation). `marlinrep2` = the same spec with that flag
removed from the no-speculation arm only; same hypotheses.
