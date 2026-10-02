# TF #179: an FP8 n-gram table declared in quantized_layers (2026-10-02)

Variant `qwen38-flash-next-mtpfp4-ple8decl`: our mtpfp4 with `layers.1.ple.ple_embedding.ngram_embedding:
{"quant_algo": "FP8"}` added to quantized_layers (config.json + hf_quant_config.json), weights hardlinked. Arms: v0.6.1
(`17c73e1`) vs v0.6.1 + #222 (`89d723e`), `tensorfold serve`, zoo_probe (3 greedy prompts).
- v0.6.1: refused at the config check (`algos - allowed = {'FP8'}`).
- #222: loads and serves sane replies — the table goes through the FP8 lane we already serve undeclared. If so, the
  reporter's two load-path refusals come from something else in their export (config_groups, the 8-bit
  group-128 MTP experts, ...), not from declaring the table.
- **Result:** as predicted. v0.6.1 refuses at the config check; v0.6.1 + #222 loads and serves (3 sane greedy
  replies, 86/99 drafts accepted). Declaring the table is fully covered by #222; the reporter's two load-path refusals
  come from other parts of their export (not identified here: their 301 FP8_PB_WO layers vs our 157, the 8-bit
  group-128 MTP experts, config_groups).
