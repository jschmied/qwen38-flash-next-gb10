GDNNQ (FUSION-AGENDA item 2, 2026-09-28 ~20:50). GDN output chain today: FLA rmsnorm_fn (gated, norm_before_gate, head
dim 128, tile [4,128], 1 warp) writes bf16 y [T, 48*128] -> QuantFP8 (group (1,128), column-major scales, CUDA
per_token_group_quant) reads y, writes fp8 A + scales -> CutlassFp8BlockScaledMMKernel GEMM (out_proj). Fused: one
Triton kernel with FLA's exact norm expression and tile, rounding to bf16 where FLA stores, then the CUDA quant formula
(absmax = max(eps, max|y|), s = absmax / 448, q = clamp(y / s) -> e4m3, divisions as div_rn), writing A and the
column-major scales directly. Head dim = quant group = 128, so each (token, head) row is one quant group.
H: A bytes and scales bit-identical to today's (so the GEMM output is identical); time at T = 3456 tokens: norm +
quant 0.54 + 0.27 = 0.81 ms today -> 0.40..0.55 ms fused (reads x and z 42.5 MB each, writes 21 MB fp8 + scales:
~106 MB vs ~212 MB), i.e. -0.26..-0.41 ms per GDN layer-chunk, x 72 per 7.5k prefill ~ -0.7..-1.1 % TTFT.
Out of range: not bit-identical (then find the rounding difference before any timing claim) or no time gain.

GDNNQ server A/B (2026-09-28 ~21:40). Overlay on the clone venv (patch_gdnnq.py, backup *.orig-gdnnq, env FN_GDNNQ=1);
installed op test with a non-contiguous z: 1 fp8 byte of 21.2 M (3,456 tokens) and of 3.7 M (595) differs, scales
identical, 6 tokens bit-exact. Arms gdnnq vs base on the full prod config (FN_HCFUSE=1, FULL_AND_PIECEWISE, K=5, bf16
state, pmu 64, KV 4 GiB), 2 starts each, probe nvprobe. The fused path also runs in the decode verify (RecoverSSM goes
through _output_projection), where it is ~2x faster per call.
H: TTFT 8k and 30k -0.7..-1.5 % (36 GDN layers x chunks x -0.39 ms); decode c=1 code/prose -0.5..-1.5 % (36 x ~-0.02 ms of
a ~64 ms cycle); c=4 -0..-1.5 %; greedy hashes may differ (drift-level op). Out of range: any cell slower by > 1 %,
or TTFT gain < 0.3 %.
