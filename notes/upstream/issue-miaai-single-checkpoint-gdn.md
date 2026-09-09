DRAFT — user go given ("write a issue to mia's repo"). MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark, new issue (2026-09-09).

TITLE: The shipped checkpoint quantises the GDN/linear-attention path with a five-month-old ModelOpt — a testable candidate for #27

## What we found

`start.sh` ships `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (98.6 GiB) because the RadixArk build
does not fit. Reading its `hf_quant_config.json` against the other public builds turns up something
that may matter for #27:

| build | ModelOpt | quantized_layers | excludes | GDN / linear-attn quantised? |
| --- | --- | --- | --- | --- |
| `local-inference-lab` (shipped here) | **0.39.0.dev290+gf9d9a71de, dated 2026-04-07** | **544** | **0** | **yes — `in_proj_a/b/qkv/z` and `out_proj`, 36 layers each, MXFP8** |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 0.46.0.dev281+g73d778422 | 50 | 292 | no |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 0.46.0 (release) | 48 expert layers | `*.self_attn.*`, `*.linear_attn.*` excluded | no |
| `primitive-ai/…-NVFP4`, `…-mixed-NVFP4-FP8` | 0.46.0 (release) | same shape as RadixArk | 13 patterns | no (FP8 in the mixed build, not NVFP4) |

The shipped build's algorithm mix is **MXFP8 × 467, NVFP4 × 48, W4A16_NVFP4 × 29**. `lm_head` and
`embed_tokens` are *not* quantised in it, so those are not the suspect.

The point is not that 544 layers is wrong — it is that **every other public build excludes the
linear-attention path entirely**, and this one quantises it, using a toolchain five months older than
the rest of the field.

## Why that is a candidate for #27, stated as a hypothesis and not a claim

#27's corruption is *ordering* damage in combining marks — a tone mark duplicated (`0E49+0E49`), a
mark emitted before the vowel it must follow (`0E48+0E35` instead of `0E35+0E48`), ASCII injected
inside a word. That is a failure of fine sequential structure rather than of token identity, and on
this architecture the sequential state lives in the GDN recurrence — precisely the path this checkpoint
quantises and the others do not.

We have one data point that the GDN path is sensitive at much smaller perturbations: changing only the
**SSM state dtype** from float32 to bfloat16 — not the projections, just the carried state — moved
**127 of 2,504 modal top-1 predictions** on a prose prompt, measured with both arms internally
bit-exact. Quantising the projections themselves to MXFP8 is a considerably larger perturbation than
that.

**Two honest caveats.** We serve the RadixArk build and have never seen this class of corruption, but
we also run no non-Latin traffic, so that is not evidence. And we have not reproduced #27 — this is a
hypothesis with a cheap test, not a diagnosis.

## The cheap test

Serve `nvidia/Qwen3.8-Flash-Next-NVFP4` or `primitive-ai/Qwen3.8-Flash-Next-NVFP4` — both exclude the
GDN path — and re-run the Thai prompts from #27. Everything else held fixed. If the corruption
disappears, the shipped checkpoint's quantisation scope is the cause and the fix is a checkpoint swap,
not a serving change. If it persists, the GDN path is exonerated and #27 is somewhere else, which is
also worth knowing and costs one server start.

Worth noting for #12 as well: `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` is already the hybrid
you describe there — FP8 on `self_attn.{q,k,v,o}_proj` and `linear_attn.{in_proj_qkv,in_proj_z,out_proj}`,
NVFP4 group-16 on the routed experts — so that enhancement may be a download rather than a build.

Happy to run any of this on our GB10 if it helps; we have the box and the deterministic serving stack,
just not the Thai evaluation.

*AI assistance was used in preparing this issue; the configuration facts were read from the published
`hf_quant_config.json` of each build and the 127/2,504 figure is our own measurement.*
