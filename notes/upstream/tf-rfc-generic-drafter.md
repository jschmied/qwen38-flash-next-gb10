DRAFT — needs the user's go. ashhart/TensorFold new issue (2026-10-10), RFC.

Title: RFC: learned drafters for any family, one block drafter in the core, trained from the engine's own recordings

#455 put an external drafter behind any `lanes.Backend`: `prepare_features` / `features` hand the drafter tapped rows of the target, and the `Drafter` vtable (`taps`, `absorb`, `hold`, `held`) doesn't know the family. What's still per model is everything around it. Our Kolibri-1 block drafter ([josch15366/Kolibri-1-block-drafter](https://huggingface.co/josch15366/Kolibri-1-block-drafter)) is Python, Kolibri-only, and was trained from recordings our Kolibri server wrote.

None of the drafter is Kolibri-specific, though. It reads three tapped layers, the target's embedding and its LM head restricted to a vocabulary slice. Its own layers (RoPE, attention, SwiGLU) don't depend on what the target is: dense, MoE, Mamba hybrid. The same training scripts trained a chain drafter and the block drafter with no model code. So the proposal is to make it the core's drafter:

1. **One block drafter in the core** (`core/drafter/block.zig`), built from the shared layers in #548. That means RMSNorm, attention and SwiGLU from item 4, and the projections from `cuda/qlinear.zig` (#587, #595), so FP8 and NVFP4 drafter weights come for free. It implements `lanes.Drafter`, and a family adds nothing beyond the `features` it already has after #455.
2. **A model-neutral release format**: `config.json` with `architecture: "block-drafter"`, the `target` repo, `taps`, the shape (hidden, heads, ffn, layers, block), the draft-vocab size and serving defaults (depth, weight formats); `model.safetensors`; `draft_vocab.json`. The embedding and the head slice ship in the release, so the drafter never reads the target's weights in whatever format they're stored. For Kolibri that's about 1 GB in bf16: the full 128k embedding (655 MB) plus a 60k head slice (310 MB). FP8 would halve it. Our Kolibri release has this shape apart from the name and those two tensors, which it takes from the target today.
3. **A recorder in the core**: on verified rows, the tapped `features` and the target's top-k (#572's logprob rows) written to disk per conversation, behind a flag. That's the training data, the same for every family; serving traffic becomes drafter data.
4. **Training stays offline in Python** (`tools/drafter/`): it reads recordings only, so it needs no model code. Per model there are two cheap steps, a tap probe (which layers carry the next token; for Kolibri 44-48 of 50) and a vocab-slice pick (target top-1 coverage per language).

What we measured on Kolibri-1 (GB10, single stream, prompts never trained on), decode speed against the engine's copy drafting alone:

| | SWE | chat | German | 4 streams |
|---|---|---|---|---|
| block drafter, greedy | 1.38x | 1.28x | 1.37x | 1.10x |
| sampled as served | 1.47x | 1.26x | 1.23x | |

The order we'd suggest: (2) and (3) first, since they're small and family-free; then (1) on the shared layers as they land; Kolibri as the first target and Nemotron second, where its own MTP head gives a baseline.

Questions for @ashhart:
- Is a learned drafter in the core something you want, or should drafters stay outside the engine and only the `Drafter` interface be core?
- Should the release carry the embedding and head slice (self-contained, about 1 GB in bf16 for Kolibri), or should the backend expose its own as a weight view?

We can take (2) and (3) as small PRs, and (1) once #548's layer pieces are in.
