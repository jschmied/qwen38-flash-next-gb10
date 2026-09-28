HCFUSE standalone (2026-09-28 ~13:00, after §5aa). One hyper-connection block at prefill shape M = 3,456 (and M = 6):
today = hc_combine_norm (writes residual + xn) -> cuBLAS down 10240->336 -> hc_silu -> cuBLAS up 320->10240 (writes
gate) -> hc_gate_mix. Fused = K1 combine that writes residual + rrms[M,4] only; K2 Triton down GEMM normalizing on
its A-load; hc_silu; K3 Triton up GEMM with the gate-mix epilogue (4 stream accumulators per tile). 531 -> ~320 MB.
H: today's block 2.4..2.8 ms at M = 3,456 (trace: 0.98 + 0.73 + ~0.5 + ~0.45); fused 1.4..1.9 ms, i.e. -0.6..-1.2 ms
per block. Correctness: residual and rrms-derived xn bit-exact to today's; block input within today's own error vs an
fp32 reference (max and mean |err| within 1.5x of today's). M = 6: correct; time not a criterion (L2-resident).
Proceed to a server A/B only if the saving is >= 0.6 ms per block at M = 3,456 and correctness holds.
Out of range: fused slower (Triton GEMM too far below cuBLAS), or block-input error > 1.5x today's.
