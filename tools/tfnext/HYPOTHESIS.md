# Next prefill lever, bound first (2026-10-02)

8k prefill profiles (torch.profiler, full kernel list) on EXL3 (#212 + prefetch, `pr212-prefetch`) and NVFP4 (#211,
checkpoint math), then ncu on the top non-expert kernels: time vs its byte / FLOP floor.
Expectations: EXL3 — experts ~30 % (down from 43 %), then DeltaNet chain, hyper-connections (_hc_* + _f16_*), attention
_chunks each 5–15 %; NVFP4 — qmmf (block-FP8 dense prompt GEMMs) ~20 %, experts ~30 %. The next lever is the kernel
furthest above its floor with ≥ 8 % of the prefill; if none is > 1.5x its floor, prefill is done for kernels.
