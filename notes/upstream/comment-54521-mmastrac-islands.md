DRAFT — needs the user's go. vllm-project/vllm #54521, reply to mmastrac (2026-09-08 16:22).

---

@mmastrac Your M table has a shape three of us have now hit independently, and there is a probe in
this thread's history you can reuse rather than rebuild.

**The islands are the interesting part.** Clean at 800–1280, 6/11 at 1536, clean again at 1792–2272,
then 10/11 at 2304 — that is not a threshold, it is a kernel-selection boundary being crossed and
recrossed. Two other data points with the same signature or its absence:

- **sm_120** (@jahnclawdmonet, earlier in this thread): non-monotonic bf16 with islands at 64, 128
  and 4096 — explicitly the one thing they could not line up against a monotonic model.
- **gfx1151 / ROCm** (@davidcanar): **monotonic**, per-shape threshold, no islands — identical
  through M=64 then differing from M=128 for the 12576×4096 and 4096×4096 KDA projections, and
  through M=32 then differing from M=64 for 3072×4096.

So: islands on NVIDIA, a clean threshold on ROCm. If that holds it points at tuned-tile selection
per M bucket rather than anything in the reduction itself.

**The probe:** `tools/gemm_m_invariance.py` in
[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10) — same
question, same output format (row 0 at M=1 vs larger M, per shape), and davidcanar contributed
`gemm_m_invariance_rocm.py` alongside it. Two warnings, both of which cost us a retracted row:

1. The **v1** script passed blockwise-FP8 scale tensors row-major where the sm_120 kernel deduces
   M-major/K-major from the problem shape, so that row never computed the intended GEMM. v2 uses
   production layouts. If you write your own, check the scale layout your kernel actually expects.
2. The MoE row means nothing unless `VLLM_TUNED_CONFIG_FOLDER` points at the deployed configs —
   otherwise the kernel silently uses the stock config and you have measured the wrong thing.
   Confirm the `Using configuration from …` log line before trusting it. This matters here
   specifically because your hypothesis is about tuned tiles.

**On "the non-determinism is not the cause of corruption":** that matches what we found separately,
and it is worth stating carefully because the two get conflated. On GB10 an independent 50-item
quality suite scored *stock* **higher** than the deterministic build (97/100 vs 95/100) while stock
had **13/50 unstable items against 0/50** — no accuracy regression, a large reproducibility one. If
your corruption is real token damage rather than run-to-run variation, non-determinism is probably
the wrong tree, and your `compute_kpool_tail_slot_mapping` and `mamba_hybrid.py` `positions` findings
look far more promising. The second in particular — tail slot mapping skipped so requests share tail
slots — is a correctness bug with an obvious corruption mechanism, and it is adjacent to #55600
(hybrid mamba prefix-cache hit reading out of bounds when `cache_config.block_size` is lowered below
`mamba_block_size`), which may be the same area.

Happy to run anything specific on GB10 (sm_121, TP=1) — we have the hardware but not your model, so
the M-invariance probe on your shapes is the part we can do for you.

<!-- AI disclosure: produced with AI assistance; I reviewed every line and ran every number quoted. -->
