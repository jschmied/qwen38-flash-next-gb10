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

GDNNQ quality screen (2026-09-28 ~22:10). §5r's evalq (GSM8K 1,319 + HumanEval 164, scored offline as the unprivileged
user), gdnnq vs base on the full prod config (HC fusion + F4), 2 starts each.
H: no measurable change: every gdnnq-vs-base McNemar p > 0.05 and both scores inside base's own start-to-start spread
(the HC-fusion screen saw base GSM8K 95.98-96.66 %, HumanEval 157-159). Out of range: a same-direction gap larger than
base's spread in both starts -> SWE-bench before any proposal.

## Agenda 2g: `venvtext` (2026-09-29, before the run) — why the warm replay is 1.8–1.9 s on the prod venv

Existing data: on the clone venv FNMOEFUSE alone leaves the replay at 1.52 s with identical greedy hashes (moefuse A/B),
GDNNQ gives 1.81 s with different hashes (combo); the replay time also moves non-monotonically with K (kcost2: K2 1.40,
K3 1.58, K4 1.42, K5 1.52, K6 1.64 s). So the replay cell is set by the 96-token reply's text (its draft acceptance).
The prod venv without GDNNQ reads 1.90 s (gqreplay). One start, prod venv, prod K5 config, FNMOEFUSE on, GDNNQ off,
nvprobe. H-text: code/prose hashes differ from the clone's d102a738 / 38c70791 (the venvs' numerics differ, so the
reply differs) and code c=1 ms/tok stays within ±3 % of the clone's 14.7. H-speed: the hashes are equal → the same text
decodes slower on the prod venv; then diff the two venvs' overlays.
