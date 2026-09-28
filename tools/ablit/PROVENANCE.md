# qwen38-flash-next-mtpfp4-ablit — provenance (built 2026-09-28 on the GB10)

Our prod checkpoint `qwen38-flash-next-mtpfp4` with the abliteration of
`dealignai/Qwen3.8-Flash-Next-ABLITERATED-NVFP4` @ `be794b990578ef3031eccf9f28e675a289a09ee9` (downloaded to the backup
box 2026-09-25, file hashes = HF's published sha256) carried over in our format. UNCENSORED research artifact (safety
refusals removed by the upstream author); same license as the base (Qwen community license).

- What dealignai changed (per-tensor diff against RadixArk/Qwen3.8-Flash-Next-NVFP4, from which both checkpoints
  derive): only 13 tensors, all BF16 — `model.language_model.layers.{3,7,…,47}.self_attn.o_proj.weight` (the 12
  full-attention layers) and `mtp.layers.0.self_attn.o_proj.weight`. Every other file is byte-identical to RadixArk.
- What this build changes relative to `qwen38-flash-next-mtpfp4` (and nothing else; per-tensor diff = 25 tensors):
  - the 12 main-model o_proj: FP8 E4M3, 128×128 blockwise, `scale = amax * (1/448)`, stored as `weight_scale_inv`.
    That recipe reproduces the FP8 o_proj of our checkpoint from RadixArk's BF16 bit-exactly (all 12 weights and
    scales, 0 mismatched bytes), so these are what the original FP8-mixed conversion would have produced from the
    abliterated BF16.
  - `mtp.layers.0.self_attn.o_proj.weight`: dealignai's BF16 copied as is (ours is RadixArk's BF16 unchanged).
- Files: all hardlinked to `qwen38-flash-next-mtpfp4` except `model-bf16-000{10,11,12}.safetensors` (rewritten copies).
  `SHA256SUMS` covers every file. Extracted source tensors: sha256 5a1aeae6…d73c (13 tensors, 408,944,640 bytes).
- Tools: speed-of-light §5ae, `tools/ablit/` in the notes repo.
