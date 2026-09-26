GDN4 (speed-of-light 5j), written 2026-09-26 before the runs. User: "ok, try it" (GDN projections as NVFP4 W4A16).
Change: the 36 GDN layers' in_proj_qkv + in_proj_z (fused in_proj_qkvz) and out_proj: FP8 block (current) -> NVFP4
W4A16 through vLLM's Marlin FP4 GEMM, quantized from the BF16 originals (RadixArk ckpt, sha256-verified extract).
Weight error: FP8 vs BF16 2.6 % (measured); NVFP4 vs BF16 ~9.5 % expected (build log prints it).
Microbench (L2 flushed): Marlin NVFP4 at its floor: qkvz 106.8 us (FP8 floor 190.7), out_proj 42.2 us (FP8 floor 71.5).
SPEED H1: per GDN layer -80 (qkvz) -29 (out) us -> ~-3.9 ms/step; in-model c1 per cycle -2.5 ... -4.0 ms
  (-4.6 ... -7.4 %), c4 +4 ... +8 %. accept_len may move a little (target outputs change): within +-0.05.
QUALITY (teacher-forced prompt logprobs, 3 long docs, lpprobe; reference = BF16 GDN projections):
  mean|dlogprob| vs BF16: FP8 base = X (small), NVFP4 = Y; expect Y ~ 3-4 X.
  NLL change vs BF16: FP8 < +0.2 %, NVFP4 +0.3 ... +1.0 %.
  NVFP4 NLL lower than BF16 is NOT a win (lower-nll-can-be-softening): read mean|dlp| first.
Decision is the user's (quality trade): report both; no prod change.
