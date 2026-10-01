POSTED 2026-10-01 (user: "post"). ashhart/TensorFold #179 comment (2026-10-01).

Supporting data for the main point. Our own Flash Next export (`mtpfp4`: NVFP4 experts, block-FP8 linears) stores the
n-gram table exactly like this checkpoint: `ngram_embedding.shard_{0..127}.weight` `F8_E4M3 [2500012, 160]` and a
`BF16 [1]` `weight_scale`. It does not list the table in `quantized_layers`, so it passes the check. TensorFold 0.6.0
has served it on one GB10 through the FP8 table lane in our exactness runs, a quant-format sweep, and a `--context
262144 --parallel 8` long mix. Every reply was sane. Drafts and resumes matched serial. So the loader side of this
request is already exercised.

The fix touches the same line as #178. That PR already narrows the check per layer and accepts FP8 on MTP expert layers
only. Accepting FP8 on `*.ple.ple_embedding.ngram_embedding` would be one more name in the same predicate. I can add it
there or send it separately after #178, whichever you prefer.

On the minor point: `ple_bytes()` is only reached through `weight_bytes()`, and that only on the MLX serve path
(`cli.py:373`). `_TABLE` is spelled for MLX conversions (`language_model.model.…`). On CUDA neither is called, so the
DGX Spark memory estimate is unaffected.

Written with AI assistance (Claude Code).
