DRAFT — needs the user's go. ashhart/TensorFold new issue (2026-10-10), RFC. Revised after review: genericity limits stated, target adapter, recipe.

Title: RFC: learned drafters for any family: one block drafter in the core, a recorder, and a recipe

#455 put an external drafter behind any `lanes.Backend`. `prepare_features` / `features` hand the drafter tapped rows of the target, and the `Drafter` vtable (`taps`, `absorb`, `hold`, `held`) doesn't know the family. Everything around it is still per model. Our Kolibri-1 block drafter ([josch15366/Kolibri-1-block-drafter](https://huggingface.co/josch15366/Kolibri-1-block-drafter)) is Python, Kolibri-only, and was trained from recordings our Kolibri server wrote.

Most of it isn't Kolibri-specific. Its own layers (RoPE, attention, SwiGLU) don't depend on the target, and the training scripts read recorded arrays, not model code. Three things are per model, though, and a shared drafter has to say so:

- **What a tap is.** Kolibri has one residual stream, and a tap is a layer's normed output. Families with several residual streams (hyper-connection lanes) or per-layer embeddings mixed into the stream have to define their tap: which stream, merged how, normed or not. `features` is the right place for that, but each family decides it once.
- **How the target turns tokens into rows and rows into logits.** The drafter uses the target's embedding and head. An embedding scale, a tied head, a logit softcap or a head bias change what the drafter must predict.
- **The design's parameters.** Block length, the chain head at position 1, the predecessor head and the spine each won or tied on Kolibri, a MoE whose extra verify rows are cheap. A dense target, or one with its own MTP head, may want other values. The code takes them as parameters, but they're measured on one model only.

The proposal:

1. **A model-neutral release format.** `config.json` with `architecture: "block-drafter"`, the `target` repo and revision, `taps`, the shape (hidden, heads, ffn, layers, block), serving defaults (depth, weight formats), and a target adapter: `embed_scale`, `tied`, `softcap`, `head_bias`. Then `model.safetensors` and `draft_vocab.json`. The embedding and the head slice ship in the release, so the drafter never reads the target's weights in their stored format. For Kolibri that's about 1 GB in bf16: the full 128k embedding (655 MB) plus a 60k head slice (310 MB). FP8 would halve it. Our Kolibri release has this shape apart from the name, the adapter and those two tensors, which it takes from the target today.
2. **A recorder in the core.** On verified rows, write the tapped `features` and the target's top-k (#572's logprob rows) to disk per conversation, behind a flag. That's the training data in one format for every family, and serving traffic becomes drafter data.
3. **One block drafter in the core** (`core/drafter/block.zig`). It's built from the shared layers in #548: RMSNorm, attention and SwiGLU from item 4, and the projections from `cuda/qlinear.zig` (#587, #595), so FP8 and NVFP4 drafter weights come for free. It implements `lanes.Drafter`. A family defines its taps in `features` and its adapter values in the release, and nothing else.
4. **Training stays offline in Python** (`tools/drafter/`). It reads recordings only, so it needs no model code.
5. **A recipe**, `docs/recipes/training-a-drafter.md`, in the form of "adding a Zig family": the steps in order, each with what it gained on Kolibri, and what didn't pay. Per model, the cheap steps are a tap probe (which layers carry the next token; 44-48 of 50 for Kolibri) and a vocabulary-slice pick (the target's top-1 coverage per language). The expensive step is days of the target serving prompts to record its own output, not code.

What we measured on Kolibri-1 (GB10, single stream, prompts never trained on), decode speed against the engine's copy drafting alone:

| | SWE | chat | German | 4 streams |
|---|---|---|---|---|
| block drafter, greedy | 1.38x | 1.28x | 1.37x | 1.10x |
| sampled as served | 1.47x | 1.26x | 1.23x | |

That's one target. We'd suggest Nemotron as the second, as the test of whether this is generic: it's a Mamba hybrid, its multi-stream question doesn't arise, and its own MTP head gives a baseline to beat or lose to.

Suggested order: (1), (2) and (5) first, since they're small and family-free; then (3) on the shared layers as they land; Kolibri first, then Nemotron.

Questions for @ashhart:
- Is a learned drafter in the core something you want, or should only the `Drafter` interface be core, with drafters living outside the engine?
- Should the release carry the embedding and head slice (self-contained, about 1 GB in bf16 for Kolibri), or should the backend expose its own as a weight view?

We can take (1), (2) and (5) as small PRs, and (3) once #548's layer pieces are in.
