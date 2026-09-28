HCFUSE standalone (2026-09-28 ~13:00, after §5aa). One hyper-connection block at prefill shape M = 3,456 (and M = 6):
today = hc_combine_norm (writes residual + xn) -> cuBLAS down 10240->336 -> hc_silu -> cuBLAS up 320->10240 (writes
gate) -> hc_gate_mix. Fused = K1 combine that writes residual + rrms[M,4] only; K2 Triton down GEMM normalizing on
its A-load; hc_silu; K3 Triton up GEMM with the gate-mix epilogue (4 stream accumulators per tile). 531 -> ~320 MB.
H: today's block 2.4..2.8 ms at M = 3,456 (trace: 0.98 + 0.73 + ~0.5 + ~0.45); fused 1.4..1.9 ms, i.e. -0.6..-1.2 ms
per block. Correctness: residual and rrms-derived xn bit-exact to today's; block input within today's own error vs an
fp32 reference (max and mean |err| within 1.5x of today's). M = 6: correct; time not a criterion (L2-resident).
Proceed to a server A/B only if the saving is >= 0.6 ms per block at M = 3,456 and correctness holds.
Out of range: fused slower (Triton GEMM too far below cuBLAS), or block-input error > 1.5x today's.

HCFUSE server A/B (2026-09-28 ~12:45). Overlay installed on the clone venv rssm (patch_hcfuse.py, backup
*.orig-hcfuse, env FN_HCFUSE=1, threshold FN_HCFUSE_MIN=512 tokens per batch). Op test: below 512 bit-identical to
today; at >= 512 drift-level (1 bf16 ulp at 512/595, identical at 3456). Arms: hcfuse vs base, prod config (PIECEWISE,
K=5, probabilistic drafts, bf16 state, pmu 64, KV 4 GiB), 2 starts each, alternating. Probe nvprobe.py (fixed replay).
H: TTFT 8k -4..-8 % (standalone -0.92 ms x ~196 blocks = -0.19 s of ~2.85 s), 30k -4..-9 %; cache-hit replay (prefill of a
short suffix + 96 tokens decode) within +-2 %; decode c=1 code/prose and c=4 within +-1.5 % (path not taken below 512);
code_c1 hashes identical if every probe prompt is < 512 tokens, drift allowed otherwise; the fused log line in the
hcfuse arm only. Out of range: TTFT gain < 2 % (the in-model block is not byte-bound the way the standalone is) or any
decode slowdown > 2 %.

HCFUSE threshold sweep (2026-09-28 ~13:10). The 512-token threshold was a guess. Agent turns prefill short suffixes
(pmu 64 leaves a few hundred uncached tokens), so the crossover matters. Standalone, the registered op with
FN_HCFUSE_MIN=1 (always fused) vs today's sequence at M = 16, 32, 64, 96, 128, 192, 256, 384, 512, 768.
H: fused loses below ~200 tokens (launch-bound, L2-resident), wins above ~300-400; crossover 200..400.
Set the threshold to the smallest M where fused wins in every size above it.

HCFUSE split-K (2026-09-28 ~13:25). The fused path's ~0.195 ms floor is K2 (down GEMM, K = 10,240) with only
cdiv(M,64) x 3 programs. Split K into SPLIT slices (each slice lies inside one HC stream when SPLIT is a multiple of 4),
write fp32 partials [SPLIT, M, 336], reduce them in a fixed order (deterministic, no atomics) to bf16.
H: with SPLIT 8..16 the fused path beats today's from M ~ 128..256 on (today 0.10..0.21 ms there), and stays at or
below the current fused time at M >= 640. Out of range: the partial write + reduce costs more than it saves (no win
below 640).

RESULT split-K sweep: fused (best split) beats today from M = 128 (−21 %) through 3,456 (−34 %, no split); loses at
64 (+8 %). Better than predicted (crossover 128..256). Op now: today's kernels below 128; split 8 below 192, 4 below
2048, none above. Op test: bit-identical below 128, 1 bf16 ulp at 128..2047, identical at 2048/3456, reproducible
run to run at every size (fixed-order reduce).
HCFUSE turn A/B (2026-09-28 ~13:40): hcfuse (split-K, min 128) vs base, prod config, 2 starts, probe
i54458/turnreplay.py (A: 20k cached prefix + N fresh tokens; B: 46 replayed SWE-bench agent turns, pure prefix
extension, max_tokens 1).
H: B turn TTFT median -2..-5 % (uncached suffixes of a few hundred tokens now take the fused path at -15..-22 % per
HC block; HC is ~14 % of prefill kernel time); A cells with >= 128 fresh tokens -2..-6 %. Recompute counts identical
between arms (same cache behaviour). Out of range: < 1 % (then short-suffix turns are dominated by something else)
or any slower cell beyond the arms' spread.

HCFUSE quality screen (2026-09-28 ~14:45, before any prod decision). The fused op is drift-level (1 bf16 ulp in the
down GEMM output / block input at some sizes; one greedy token flipped in 46 replayed turns). Screen = §5r's evalq:
GSM8K test (1,319) + HumanEval (164, scored offline, McNemar per pair), thinking off, greedy, c = 16, prod config on the
clone venv (K=5, probabilistic drafts, bf16 state, pmu 64, KV 4 GiB); arms hcfuse vs base, 2 starts each.
H: no measurable quality change: both scores inside the base arm's own start-to-start spread (§5r prod spread:
GSM8K 95.91-96.44 %, HumanEval 158-159), McNemar p > 0.05 for every hcfuse-vs-base pair. Out of range: a gap larger
than the base arm's own spread in the same direction in both starts -> SWE-bench before any prod use.
