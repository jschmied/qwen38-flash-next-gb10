B12X-GATED at intermediate 640 (TODO item 6 step 1), 2026-09-28 ~14:10, before the run.
The b12x gated-optimized MoE kernel (fused FC1 -> SwiGLU -> FC2-input quant) is gated off for I > 512 by
_GATED_OPTIMIZED_MAX_INTERMEDIATE_SIZE = 4 x 128. The kernel itself groups gate tiles in chunks of
_TASK_SLICE_CHUNK = 4 and holds "retained Q1 slices" partly in registers, so 5 slices (I = 640) may or may not work.
Standalone, random NVFP4 weights at the model's shapes (E 512, top-10, H 2560, I 640), same inputs to both:
generic kernel (today's b12x choice) vs gated with the guard raised to 640, at M = 3456 and M = 64.
H0 (fits): gated compiles and runs, output within FP4-requant noise of generic (relative L2 error < 2 %), and faster
at M = 3456 by 10..30 % (the activation round trip is ~8 % of MoE prefill bytes; the gated path also skips the
generic path's routed-rows workspace). H1 (does not fit): compile error, wrong output (error >> 2 %) or a crash;
then the CUTLASS EVT route. Either outcome is recorded; nothing touches a server.

VOID run 1 (2026-09-28 ~14:20): flashinfer's _DYNAMIC_KERNEL_CACHE key does not include the gated decision, so the
"gated" wrapper reused the generic kernel compiled before the guard was raised (equal times 9.02 / 8.98 ms; the 0.7-0.9 %
difference is b12x run-to-run variation). v2: one process per mode, guard raised before anything compiles, witness =
the class of every cached dynamic kernel, plus a same-process repeat to measure the nondeterminism floor.
