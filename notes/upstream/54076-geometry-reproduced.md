POSTED 2026-09-22 — vllm#54076 comment
<https://github.com/vllm-project/vllm/pull/54076#issuecomment-5779194753>

@MaCoredroid — ran your config. **It reproduces, and my 2026-09-20 comment here was wrong.** I said
"it cannot be produced on this build"; it can, on that same build, via the path you gave.

**Box/build**: DGX Spark GB10, sm_121, TP1. vLLM `0.28.1rc1.dev524+g5db652225` (base 2026-09-08) —
the same build my negative result came from. Model: Qwen3.8-27B-FP8 (`model_type: qwen3_5_text`,
hybrid linear+full attention, 64 layers, fp8 blockwise `[128,128]`). Your config verbatim:
`--enable-prefix-caching --mamba-cache-mode align`, `extract_hidden_states` with
`num_speculative_tokens=1` and `eagle_aux_hidden_state_layer_ids=[32]`,
`ExampleHiddenStatesConnector` as `kv_producer`.

Both lines come out identical to your 0.28.0 excerpt:

```
Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.
Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block
```

Same 800/200, same 1,228,800 bytes, same 37.50 %. The server booted and answered a request.

**Why my negative was wrong.** I probed the divergence by forcing `--block-size` on a different
model (Qwen3.8-Flash-Next). On *that* path the mechanism I described is real — `interface.py:931` is
a floor, so an explicit `816` is raised to `1568`, and `:957` pads the mamba page to be exactly
equal. But that is one route, not the only one, and I wrote the conclusion as a property of the
build. The hidden-state cache layer reaches the split by a different mechanism entirely:
`cache_only_layers.64` is sized at 200 against the mamba page at 800, with no `--block-size`
involved. Sorry for the noise — the earlier comment should be read as scoped to that one arm.

**What this does not answer.** My build is **777 commits behind current main** (`1ea7c63f4`); your
static reference `382970ee6c` is 38 behind. So this establishes that the divergence **persisted past
0.28.0 through 2026-09-08**, not that main still has it. If a current-main run is what you need, say
so — I have the aarch64 nightly wheel for `1ea7c63f4` on the box but have not stood a runnable
environment on it yet.

**One reproduction note, in case it saves someone time.** `eagle_aux_hidden_state_layer_ids` is not a
`SpeculativeConfig` keyword — neither on dev524 nor on current main (checked the `1ea7c63f4` wheel);
`config/speculative.py` reads it off `draft_model_config.hf_config`. Passing it at top level fails
pydantic validation. The form that works:

```
--speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,
  "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[32]}}}'
```
