# Closed levers and capability traps

<!-- moved out of README.md on 2026-09-05 so the README stays an entry point; content unchanged -->

## Two things that decide whether the model is *usable*, and no speed test can see

Found on 2026-08-29, after a full day of throughput work:

- **Tool calling was rejected outright.** Our launcher set `--reasoning-parser qwen3` and nothing
  else, so every request carrying `tools` returned **HTTP 400**. Fixed with
  `--enable-auto-tool-choice --tool-call-parser qwen3_xml`; now **32/32** across temperatures
  0.2 / 0.6 / 1.0 / default. [Write-up](notes/tool-calling-was-off.md).
- **`--max-model-len 8192` could not hold the model's own reasoning.** A code task emitted 31,115
  characters of thinking before 12,931 of content. 8192 was chosen for benchmarking and was never
  going to serve real work.

Neither is visible to throughput, acceptance, NLL, divergence or coherence tests, because none of
those sends a `tools` field or a long generation. **A serving config has capabilities, not just
speed** — probe both before benchmarking a new recipe.

## What we would not try again

The most useful half of this repo. Each of these looked like a lever and measured null, with the
mechanism understood rather than shrugged at:

- **Hyper-connection quantization or kernels.** 27% of decode GPU time, three interventions
  (blockwise FP8, the CUTE-DSL skinny GEMM, per-channel FP8), all null. They are **latency-bound
  at ~78% of roofline** — a quarter of decode time because there are ~102,000 of them, not because
  any one is expensive. Corroborated independently three ways.
  [Why](notes/why-the-hyper-connections-do-not-respond.md).
- **NVFP4 KV cache** — closed by two independent GB10 measurements plus a structural
  MTP-acceptance penalty, and it fails silently.
- **Lowering `gpu-memory-utilization` to avoid host freezes** — refuted; 0.70 is the worst
  recorded outcome. The cause is absolute free memory at launch, not the ratio.
