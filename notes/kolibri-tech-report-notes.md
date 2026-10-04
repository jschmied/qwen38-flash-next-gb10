# Kolibri-1 tech report: what matters for us

Source: Aleph Alpha, Kolibri-1 technical report (https://aleph-alpha.com/downloads/tech-report.pdf, 189 pages), read
2026-10-04. Page numbers are the report's.

## Drafting
- No MTP, speculative decoding or draft head anywhere in the report, and none planned (future work, p95).
- Policy entropy is very low after RL (normalised 0.033, p87): the acceptance ceiling should be far above our 65 %
  first-step agreement, so the gap is likely our drafter or its data (our inference).
- SFT teachers were GLM-5.2, GLM-5.3 and Qwen3.8-27B (p53); RL (1000 steps, 36 % SWE + terminal, temperature 1, top-p 1,
  effort mostly high, p74/p76/p183) moved the model from them. Kolibri's own temperature-1, high-effort SWE and terminal
  generations are the closest match to what it serves: the right drafter data.

## Architecture (our engine matches)
- Sliding window: the 512 preceding tokens and the query (p7). RoPE theta 10,000 on the 40 sliding layers only; full
  layers have no positions; no RoPE scaling at any length (p8, p12). Full-attention layers by config: 4, 9, ..., 49
  (so our taps 44 and 49 are full-attention layers, 47 sliding).
- Per-head q/k RMSNorm with learned gains before RoPE, standard 1/sqrt(128) scaling, no sinks or soft-capping (p8, p134).
- Sandwich norms before and after both sublayers; post-sublayer gains initialised to 1/sqrt(50) (p19).
- Routing: top-6 of (logit + bias), weights unbiased sigmoid(logit), no renormalisation, no routed scaling, ungated shared
  expert; bias centred (p16-17). Every layer MoE, expert width 512.
- Layers 0 and 1 barely use their routed experts (bias overrides 98.5 % of choices; silencing them changes loss ~0,
  p138-140): a routing bug there would not show in quality checks — test the bias path elsewhere.
- LM-head logits in fp32 in trainer and inference (p180, p182); router, embeddings, norms bf16.

## Numerics they trained for
- RL was FP8 quantisation-aware: E4M3 weights in 128 x 128 blocks with fp32 scales, dynamic activation quantisation in
  1 x 128 groups, and Q/K/V rounded to FP8 with unit scales (FP8 KV cache, no calibration) (p74, p180). QK-norm keeps
  Q/K/V below 448, so unit-scale FP8 KV cannot overflow (p12).
- So the checkpoint's own math is FP8 activations (1 x 128 groups) and an FP8 KV cache. Our engine runs bf16 activations
  and a bf16 cache: more precise than what the model was trained on, and a candidate opt-in "checkpoint math" mode
  (FP8 prompt GEMMs, half the KV bytes) — a separate conversation under TensorFold's rules.
- No BF16-vs-FP8 benchmark comparison in the report.

## Serving and evaluation
- Release sampling: top-p 0.97, top-k 128, temperature from the generation config (1.0); reasoning effort high by default
  (p90, p164). Effort mapping: none/enable_thinking=false -> none; minimal/low -> low; medium -> medium;
  high/xhigh/max/default -> high (p50-51). In "none" mode the model writes an empty reasoning block (p64).
- SWE-bench Verified 66.4 with OpenCode, <= 250 steps, effort high, web fetch/search removed (with them it found upstream
  fixes in ~7 % of trials, ~+7 pp) (p93-100). Development curves used mini-swe-agent (p90). Our mini-swe-agent runs at
  temperature 0.6 / top-p 0.95 / top-k 20 are not comparable to their number.
- Throughput numbers are vLLM, FP8 weights and KV, 8 x B200 (p130); no single-GPU or GB10 numbers.
- Long context: trained to 256k, RULER to 1M without rescaling (p41, p48: 128k 67.9, 256k 69.8, 1M 63.2).
