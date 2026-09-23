POSTED 2026-09-23 — vllm#54076 comment (v2, after two deletions)
<https://github.com/vllm-project/vllm/pull/54076#issuecomment-5789112191>
Fixes vs v1: no PLE claim (the 27B is qwen3_5, it has none — finding 221); shadowing justified
by the SHARED worker/executor/loader files; '1ea7c63f4 = main as of 09-22, since moved 39
commits' instead of 'current main HEAD'; the 200-token plateau reported as an observed match
to the logged hidden-state block size, not as cache_config.block_size, which I could not
substantiate from the retained logs.

Ran the nightly, and also the prefix-cache cell @wickist accepted on 09-16. Three results.

**1. The startup geometry persists on main.** Build `0.29.1rc1.dev533+g1ea7c63f4` — main as of
2026-09-22; main has since moved 39 commits, so read this as that revision, not as today's HEAD.
GB10 sm_121, TP=1, Qwen3.8-27B-FP8, your config verbatim.

```
Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.
Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block
```

Identical across three revisions: your 0.28.0, my `0.28.1rc1.dev524` (2026-09-08), and this one.

Run as stock upstream: the nightly wheel extracted and placed on `PYTHONPATH` so my own patched
`vllm` in site-packages is shadowed. That matters here because my patch set includes shared
worker/executor/loader files — `v1/worker/gpu_worker.py`, `v1/worker/gpu/model_runner.py`, both
`v1/executor/*_executor.py`, `envs.py`, `config/parallel.py`, `vocab_parallel_embedding.py`,
`weight_utils.py` — which are on any model's path. The runner logs the resolved `vllm.__version__`
*and its path* before starting, so a silent shadowing failure would print `dev524` rather than
quietly yielding a fake "main" result; it resolved to the wheel.

**2. Runtime: it generates.** Serving state after 400 s; `finish_reason: length`, 40 completion
tokens, coherent text, `system_fingerprint: vllm-0.29.1rc1.dev533+g1ea7c63f4`. The response carries
`kv_transfer_params.hidden_states_path` pointing at a written `.safetensors`, so the
`ExampleHiddenStatesConnector` producer path actually ran rather than merely being configured. That
closes the "runtime correctness untested" caveat for this config.

**3. The prefix-cache cell: the patch is a no-op here, on both axes.** Your diff rebased to `dev524`
(3 hunks, 1 file; hunk 1 dropped as already present, hunk 4 in your 09-06 form) applies clean. The
patch was toggled per arm and the marker `mamba_state_block_sizes` verified **0 on every unpatched
arm and 4 on every patched arm** immediately before each measurement.

Hit rate — 2 arms x **3 starts**, from `vllm:prefix_cache_hits_total` deltas (`usage.cached_tokens`
reads 0 on real hits on this build, so it cannot be used):

| arm | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| unpatched x3 | q238 h0 | q238 h200 | q238 h200 |
| patched x3 | q238 h0 | q238 h200 | q238 h200 |

Byte-identical. Hits plateau at exactly **200** tokens from pass 2 on, which matches the block size
the startup log reports for the hidden-state cache layer; a 238-token prompt leaves a 38-token
remainder. (I did not capture the engine's resolved `cache_config.block_size`, so I am reporting the
coincidence of the numbers, not asserting the mechanism.)

Output equivalence — because a hit *count* can match while the resumed state is wrong, which is what
the PR guards against. Temperature 0, fixed seed, 2 arms x 2 starts x 3 passes: **one output hash
across all 12 passes**, cold pass (0 hits) byte-identical to the resumed passes (200 hits), patched
and unpatched.

**So on this configuration the resume is already exact without the patch, and the patch perturbs
nothing.** Scope: one prompt shape, one model, temperature 0 — this shows the failure does not
manifest here, not that the fix is unnecessary. That is consistent with your defence-in-depth framing
for explicit `--block-size` and other page layouts, and those are the layouts I cannot reach from the
CLI on this model. A layout with more than one mamba group, or a prompt spanning many blocks, is
untested by this.

One reproduction note: `eagle_aux_hidden_state_layer_ids` is not a `SpeculativeConfig` keyword on
`dev524` or on main — `config/speculative.py` reads it off `draft_model_config.hf_config`, and passing
it at top level fails pydantic validation. The form that works:

```
--speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,
  "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[32]}}}'
```
