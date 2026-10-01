# Flash Next routed NVFP4 experts in the checkpoint's math (FP4 x FP4) on TensorFold 0.6.1

User 2026-10-02: "I would just post proposed pr" (TF #173 closed with 0.6.1's `--precision checkpoint`, which reaches
the 27B only; Flash Next's routed experts still run `nvfp4_expert_kernel` on bf16 rows).

Design: rows quantized once a layer under gate/up's input scale (uniform over the 512 experts in our checkpoint; the
max otherwise, as FlashInfer's CUTLASS MoE does); a grouped warp kernel on the plan's items, FP4 mma m16n8k64 against
the 27B's pack4 words + block scales per expert; gate|up epilogue = SwiGLU in fp32, quantized to down's NVFP4 input
under each expert's own down input scale; down writes each pair's row. One K chain a pair: bits never depend on the
other pairs (drafted == serial, resumed == fresh hold).

Predictions (one GB10, our ModelOpt checkpoint `qwen38-flash-next-fp8head`, 0.6.1 + #180 not needed):
- Exactness: the kernel matches a torch reference of the same math (quantized rows, dequantized weights, fp32) to
  fp32 accumulation-order noise (rel < 1e-5 on pre-quantization outputs); bits independent of the other pairs.
- Prompt 8k: experts were 36.7 % of 6.11 s on 0.6.0 (2.24 s); the DRAM floor of reading all experts once a chunk is
  ~1.1 s. Prediction −8…−18 % prefill at 8k and 32k. Below −5 %: the kernel is not the fix as built.
- Decode c=1: within ±5 % (bandwidth-bound either way).
- Quality vs `--precision full`: top-1 agreement drops 1–4 pp, like the 27B's checkpoint vs full (98.9 → 92.9 % is
  for every layer; here only the routed experts change, so smaller).
- **Result (§5bd):** exactness as predicted (bytes vs a torch quantizer < 1 % edge flips, products 1e-5, pair bits
  invariant). Prefill −8.7…−9.9 % at 8k and 32k (in range, low end; experts −26 %). Decode −0.6 % over 8 prompts (in
  range; one reply alone showed −11 % from acceptance). Quality vs full: KL 0.038, top-1 95.6 %, ppl +1.27 % (top-1
  out of range: −4.4 pp, wikitext 93.0 %).
