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
