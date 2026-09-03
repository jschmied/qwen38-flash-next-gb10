Thank you — that is the validation this PR needed, and the artefact repo makes it reproducible.

Two notes on your findings, both agreeing with yours:
- The 3/4 case (first request a partial prefix-cache hit, logits differ from the full-hit path, text unchanged) is the align-resume path: vllm#54076 / #53798 change the block-boundary seeding for hybrid models; with both applied on our box the partial-hit and cold paths agree. Independent of this kernel, as you say.
- Batch invariance: correct, nothing here changes it; the GDN path has no batch-invariant mode.

The MoE data point (no fused-finalize divergence on your shapes) is useful for #54945 — on our shapes it showed only in the decode step at M ≤ 52 tokens, never at M = 55, so it is shape-gated.
