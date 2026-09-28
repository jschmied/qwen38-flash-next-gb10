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
