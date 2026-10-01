DRAFT — needs the user's go. ashhart/TensorFold #178 reply to "Which export did you measure on? We'd like a real
receipt for it too." (2026-10-01).

The export is public: [lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8](https://huggingface.co/lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8)
at `a9786bf`. Its MTP experts sit in `model-bf16-00011.safetensors` (sha256 `f2b54563…ace819`). They're per-tensor FP8
E4M3 with a scale per tensor, and its `quantized_layers` doesn't list them. Our local copy shares 205 of 206 weight
shards with it byte for byte, by their published hashes, and differs in one BF16 shard of ours.

On one GB10 with that published layout, three greedy prompts:
- **0.6.0:** passes the check, then fails at load with `KeyError: 'mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_2'`
  (the FP8 drafter experts are read as NVFP4).
- **This PR:** loads, gives the same three replies as with an NVFP4 drafter, and accepts 86 of 93 drafts (86 of 99
  with the NVFP4 drafter).

So that checkpoint doesn't load on 0.6.0 as published, and the PR's dtype detection is what makes it work. Declaring
the two `mtp…experts` entries as `FP8` in the config (what we first tested) takes the same path once the check allows it.

Written with AI assistance (Claude Code).
