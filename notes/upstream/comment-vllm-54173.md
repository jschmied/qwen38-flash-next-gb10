# Comment for vllm#54173 (post AFTER the new issue has a number; replace NNNNN)

Data point from the same build (`0.1.dev20073+g8e685d198`), same hardware (GB10, sm_121) and the
same checkpoint, but a different symptom, so please read it as a cross-reference, not a diagnosis:

- We do not hit the illegal memory access here (`--max-model-len 8192`, prompts ≤ 4k tokens,
  MTP n=5, `--mamba-cache-mode align`), so no claim about your crash.
- What we did find and fix on this build is that the FlashInfer CUTLASS NVFP4 MoE returns
  different logits for identical requests at temperature 0; cause is the fused finalize's atomic
  reduction, `use_fused_finalize=False` makes it bit-stable at +3.6 % decode. Details and
  before/after in #NNNNN.
- One thing in your report lines up with what we measured: turning the prefix cache off changes
  more than the cache. With it on, the block-aligned mamba split runs a 55-token prompt as two
  chunks (52 + 3); off, it runs as one. So "cache off = no crash" also means "different MoE
  shapes", which is worth keeping in mind when bisecting.
- Cheap discriminator if you want one: `--moe-backend emulation` takes the NVFP4 MoE kernel out
  entirely (dequantised experts; +17 % decode here). If the crash survives that, the MoE kernel
  is excluded for your fault as well.
