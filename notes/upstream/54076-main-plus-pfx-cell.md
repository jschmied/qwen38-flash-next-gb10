DELETED BY THE USER 2026-09-23 — second deletion. Reason: the post justified 'stock main' by
PLE-offload shadowing, but qwen38-27b-fp8 is qwen3_5 and has NO PLE (finding 221). The
measurements stand; the methodology paragraph was wrong. Do not repost without fixing that
paragraph AND getting a fresh go.


Ran the `1ea7c63f4` nightly, and also the prefix-cache cell @wickist accepted on 09-16. Three results.

**1. The startup geometry persists on current main.** Build `0.29.1rc1.dev533+g1ea7c63f4`
(= upstream/main HEAD), run stock: the nightly wheel extracted and put on `PYTHONPATH`, so my own
PLE-offload patches in site-packages are shadowed. The runner logs `vllm.__version__` *and its path*
before anything else, so a silent shadowing failure would show `0.28.1rc1.dev524` instead of quietly
producing a fake "main" result; it resolved to the wheel. GB10 sm_121, TP=1, Qwen3.8-27B-FP8, your
config verbatim.

```
Setting attention block size to 800 tokens to ensure that attention page size is >= mamba page size.
Using block size 200 for hidden-state cache layer cache_only_layers.64; page alignment wastes 1228800 bytes (37.50%) per block
```

Identical across three revisions now: your 0.28.0, my `dev524` (2026-09-08), and main `1ea7c63f4`.

**2. Runtime: it generates.** Server reached serving state after 400 s; `finish_reason: length`,
40 completion tokens, coherent text, `system_fingerprint: vllm-0.29.1rc1.dev533+g1ea7c63f4`. The
response also carries `kv_transfer_params.hidden_states_path` pointing at a written `.safetensors`,
so the `ExampleHiddenStatesConnector` producer path actually ran rather than merely being configured.
That closes the "runtime correctness untested" caveat for this config.

**3. The prefix-cache cell: the patch is a no-op here, on both axes.** Your rebased diff (3 hunks
against `dev524`; hunk 1 dropped as already present, hunk 4 in your 09-06 form) applied clean. Patch
toggled per arm, with the marker `mamba_state_block_sizes` verified **0 on every unpatched arm and 4
on every patched arm** immediately before each measurement.

Hit rate — 2 arms x **3 starts**, `vllm:prefix_cache_hits_total` deltas (`usage.cached_tokens` is
inert on this build):

| arm | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| unpatched x3 | q238 h0 | q238 h200 | q238 h200 |
| patched x3 | q238 h0 | q238 h200 | q238 h200 |

Byte-identical. `200` is exactly one block — `cache_config.block_size` is 200 here — so a 238-token
prompt caches one block plus a 38-token remainder and plateaus from pass 2.

Output equivalence — because a hit *count* can match while the resumed state is wrong, which is what
the PR actually guards against. Temperature 0, fixed seed, 2 arms x 2 starts x 3 passes:
**one output hash across all 12 passes**, `cold == resumed` true in every arm, patched and unpatched.

**So on this configuration the resume is already exact without the patch, and the patch perturbs
nothing.** Two things worth saying about scope. This is one prompt shape on one model at temperature
0; it shows the failure does not manifest here, not that the fix is unnecessary — which is consistent
with your own defence-in-depth framing for explicit `--block-size` and other page layouts, and those
are the layouts I cannot reach from the CLI on this model. And a layout with more than one mamba
group, or a prompt spanning many blocks, is untested by this.

One reproduction note for anyone else running it: `eagle_aux_hidden_state_layer_ids` is not a
`SpeculativeConfig` keyword on `dev524` or on main — `config/speculative.py` reads it off
`draft_model_config.hf_config`, and passing it at top level fails pydantic validation. The form that
works:

```
--speculative-config '{"method":"extract_hidden_states","num_speculative_tokens":1,
  "draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[32]}}}'
```
