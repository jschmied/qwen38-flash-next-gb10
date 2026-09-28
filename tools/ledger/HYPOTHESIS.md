LEDGER (TODO 5b), 2026-09-28 ~12:35, before the capture. User: "do we have fusable kernels where a fusion would lower
bytes read/written? I think we checked only for launch overhead".
Instrument: one nsys trace (cuda-graph node level) on the prod config (clone venv rssm, PIECEWISE, K=5, probabilistic
drafts, bf16 SSM state, prefix-match-unit 64, KV 4 GiB): one ~7.5k-token prefill (2 chunks of 4096) and a 64-token
decode window, both after a warm-up. GB10's ncu has no DRAM byte counters (finding 144), so bytes come from shapes
(hidden 2560, 4 hyper-connection streams, 48 layers, MoE 512 experts top-10, intermediate 640, GDN 16/48 heads x 128).
Per kernel family: time, bytes it must move (weights + in/out activations), producer->consumer pairs whose intermediate
exceeds the 24 MiB L2 (so it round-trips DRAM), and the fused lower bound.
H (prefill, per 7.5k prefill): intermediates that round-trip DRAM are 10..25 % of all prefill bytes. Largest:
 (1) hyper-connection streams (4 x 2560 x tokens, read+written around every layer): 3..6 % of TTFT;
 (2) MoE GEMM1 output -> SwiGLU -> fp4 quant -> GEMM2 (0.19 GB/layer at 7.5k): 5..9 % of TTFT (finding 144: ~1.1 of
     13.5 ms per MoE layer);
 (3) FP8 act-quant re-reading bf16 activations: 1..3 %.
 Total byte-fusion prize at prefill 8..15 % of TTFT, most of it in (1)+(2).
H (decode, 6-row verify): intermediates < 1 % of bytes; only state-sized ones matter (RecoverSSM commit -> next verify
checkpoint re-read, ~0.25 ms/step at bf16). Byte-fusion prize at decode < 1 %.
Out of range: prefill intermediates < 5 % (then byte fusion is not a lever and the TODO item closes) or > 30 %.
