POSTED 2026-10-01 as https://github.com/ashhart/TensorFold/issues/173 — user's go 2026-10-01 ("post whats ready"). ashhart/TensorFold issue (question).

TITLE: Would you take an opt-in FP4-activation prompt path for NVFP4 experts, like --prefill-fp8?

Before building anything I'd like to know whether this fits TensorFold's rules.

**Where Flash Next's prompt time goes on CUDA.** One cold 8k-token prompt, 0.6.0, one stream, one DGX Spark (GB10),
our ModelOpt checkpoint (NVFP4 routed experts, block-FP8 dense layers), nsys: 6.11 s wall, 96 % of it in kernels.
The routed experts' `nvfp4_expert_kernel` takes 36.7 % of GPU time, the block-FP8 dense matmuls (`qmmf_kernel`)
18.5 %, `_b16mm` 7.2 %, attention 4.9 %. vLLM serves the same checkpoint's 8k prompt in 2.6 s.

**It is not batching.** Prompt chunks of 2,048 / 4,096 / 8,192 rows give the same TTFT within ±2.4 % at 8k and 32k
(greedy replies byte-identical), and on an EXL3 pack the routed-expert window at 512 / 1,024 / 2,048 rows moves it
−2…+5 %. The cost is per row at the experts' arithmetic: bf16 activations on bf16 MMAs, where vLLM runs these experts
W4A4 on FP4 tensor cores.

**`--prefill-fp8` shows the shape.** On the same checkpoint, FP8 prompts are 1.14–1.19× faster than bf16 (8k:
5.19–5.25 vs 6.00–6.08 s; 32k: 21.4–22.1 vs 25.1–25.5 s; three starts), but only the dense projections switch.

**The proposal:** an opt-in prompt mode for NVFP4 routed experts that quantizes the expert inputs to NVFP4 (e2m1 with
an e4m3 scale per 16 values and the checkpoint's own `input_scale`) and runs the expert GEMMs on FP4 tensor cores.
Off by default; prompts only, decode unchanged; the stored weights are used as they are. Scales are per row, so a
row's bits still don't depend on how many rows share a pass, and every pass on a server uses the same precision, so
drafted == serial and resumed == fresh hold as with `--prefill-fp8`.

What I know about the cost, none of it on TensorFold:
- On Qwen3.8-27B with vLLM, our W4A16 checkpoint (Marlin) against Unsloth's W4A4 one (CUTLASS; the two share an
  identical `lm_head` but are separate quantizations, so this is not a clean activation-only A/B), 24 agent contexts:
  W4A4 cut TTFT 10.0 → 6.4 s and lost a little more agreement with BF16, top-1 −1.53 pp vs −1.11 pp (paired, t = 2.90).
- The RadixArk NVFP4 Flash Next checkpoint served by vLLM (W4A4 experts) shows combining-mark damage in Thai at
  sampling temperature (at least 5.68 per 1,000 Thai characters in our probe; llama.cpp on the same weights reports
  none, vllm#54739); we never isolated whether activation quantization contributes.

If this is a direction you'd take, I'd build it behind a flag and bring TTFT at 8k / 32k, your teacher-forced NLL
against bf16 prompts, and the Thai probe. If not, I'll leave it.

Written with AI assistance (Claude Code).
