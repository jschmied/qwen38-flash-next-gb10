# BF16 small-M GEMM microbench (written 2026-09-25 ~01:20, before the run)

Context (40-prof, step 3): the target's BF16 linears (hyper-connection mixers down [324x10240] / up [10240x320], shared
expert gate_up [1280x2560] / down [2560x640], router [512x2560], in_proj_ba [96x2560]) run in prod on cuBLAS
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x{1,2}` (32-thread blocks). In-model per call: 40.5 / 33.9 /
46.0 / 23.2 / 25.6 / 17.2 us vs byte floors 31.3 / 29.8 / 29.8 / 14.9 / 11.9 / 2.2 (220 GB/s). ~13 ms/step on main.

Standalone, M=4, weights rotated over >= 96 MiB so L2 never holds them, a CUDA graph of back-to-back calls:
- H1: torch F.linear (default cuBLAS) reproduces the in-model per-call times within +-15 % -> the kernel itself is
  the cost, not in-model contention. Outside: at the floor standalone -> the in-model excess is contention/ordering.
- H2: at least one alternative (cuBLASLt via preferred_blas_library, FlashInfer mm_bf16 / tinygemm_bf16, a Triton
  split-K GEMV) reaches <= 1.10x floor on the mixer down and router shapes. Expected saving if swapped in-model:
  2-4 ms/step (4-7 %). Outside: nothing beats cuBLAS by > 5 % -> BF16 lever is only the 8-bit weights (-8 %).

## Server A/B (45-bf16sk, written 2026-09-25 01:12, before the run)

Deterministic two-pass kernel (`fn_bf16sk.py`), standalone <= 1.06x floor on all six shapes at M=1/4/16. Replacing
cuBLAS in-model removes ~1.0 ms (mixer down) + ~0.6 ms (mixer up) + ~0.5 ms (in_proj_ba) + the ~190 splitKreduce
launches per step; shared/router gains are uncertain (contention-bound in-model).
- Expected c=1 decode: +2.5 % to +5 % tok/s (sk vs base), every start of sk above every start of base.
- c=4: +1 % to +4 %.
- Acceptance within +-2 pp (numerics change, drafter shares the kernel); within-arm output hashes identical across
  starts (the kernel is deterministic); cross-arm hashes are EXPECTED to differ (different reduction order).
- Outside: < +1 % -> the in-model win is eaten by contention/launch cost; > +6 % -> something else changed, check
  the path lines.

## Profile with FN_BF16SK=1 (47-profsk, written 2026-09-25 02:3x, after sk0/sk1 showed a per-cycle null at c=1)

A/B so far: c=1 per verify cycle base 59.1 ms vs sk 59.4-59.9 ms (null or slightly worse), acceptance 51.2 -> 48.7 %,
c=4 +3..5 %. The kernel swap should have removed ~2 ms/cycle. Candidates:
(a) the Triton kernels run slower in-model than standalone (mixer down expected ~31 us; if >= 38 us, in-model
    conditions such as L2 state, launch cost, the partials buffer or graph capture eat the gain);
(b) the kernels are fast, but the time moved elsewhere (e.g. the MoE aux-stream overlap changed, or the drafter's M=1
    path got slower);
(c) the lower acceptance means more drafter + verify work per accepted token (already folded into ms/cycle, so it
    cannot explain a per-cycle null).
Expected: per-cycle BF16 category <= 12 ms (was 14.6 + splitKreduce) if (b); >= 14 ms if (a).

## Clock check (written 2026-09-25 ~03:3x, after 47-profsk)
47-profsk: Triton mixer down 41.1 us in-model (never overlapping a side stream) vs 30.3 standalone and ~34 in the idle
in-worker bench (2c). H: under sustained c=1 decode the SM and/or memory clock or the power cap drops >= 15 % below
the short-burst level -> the whole-step slowdown vs benches is clock/power, not the kernels. If the clocks during
decode equal those during a microbench burst, the cause is elsewhere (CPU DRAM traffic on the unified memory,
TLB/page layout of the model weights).

## Producer->consumer micro-chain (pairbench, written 2026-09-25 ~03:35, before the run)
Consumer: mixer down [336x10240] (Triton det 2-pass and cuBLAS), M=4, weights rotated. Producers before each call:
P0 none, P1 tiny elementwise ([4,10240] add, kernel boundary only), P2 fp32 write of 3 MiB (GDN-state-sized per layer),
P3 fp32 write of 12 MiB, P4 read-only 12 MiB reduction. Marginal consumer cost = graph(P+C) - graph(P alone).
- (a) dirty-L2 write-back: P2/P3 raise the consumer's marginal cost by >= 5 us (P3 > P2), P4 does not.
- (b) kernel-boundary ramp: P1 raises it by >= 5 us over P0.
- Neither -> the in-model overhead needs the real sequence (in-worker replay).

## Bimodality check (written ~03:40 after pairbench showed P0 42.1 vs P1 30.8 us for the same consumer)
H: the same mixer-down chain repeated 40x is bimodal, ~30 and ~41 us (ratio ~1.33 = LPDDR5x 8533/6400 MT/s bins),
independent of code -> the "in-model excess" is a DRAM frequency/power state. Unimodal -> pairbench noise is
something else.
