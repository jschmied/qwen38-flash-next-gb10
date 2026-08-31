# The MoE backend axis: closed, and what it cost to close it

**Verdict.** `flashinfer_b12x` is *selectable* on this hardware and *faults*: an illegal memory
access on sm_121, which is a known unfixed upstream bug (**vllm#50189**, open since 2026-07-28 —
their trigger is concurrent chunked prefill, ours is kernel load with no traffic; we added the
second reproduction). `auto` picks `FLASHINFER_CUTLASS`, which works. **Nothing further to try here
without an upstream kernel fix.**

**The blocker we spent weeks on was not the real one.** This file claimed since August that
*"`--moe-backend` is global, so the drafter's unquantized MoE vetoes the kernel for all 48 quantized
layers."* The first half is **false**: `SpeculativeConfig.moe_backend`
(`config/speculative.py:118`) sets the drafter's backend independently, and its docstring names our
exact case. Found via **vllm#51960**, whose author hit the identical trap. The supported config was
always:

```
--moe-backend flashinfer_b12x \
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"flashinfer_cutlass"}'
```

⚠️ **…except that field is inert on the V2 runner, which we measured and reported.** The engine
accepts it and echoes it back in its config dump, then fails with the original error. It is read at
`v1/spec_decode/llm_base_proposer.py:1295` (V1 proposer) and **nowhere** in
`v1/worker/gpu/spec_decode/` — and #53896's own commit *"Limit Qwen3.8-Flash-Next to model runner
V2"* pins this model to the path that ignores it. Reported on #51960 before their error-message
change lands, since it would otherwise send users to an option that silently does nothing.

## The one durable win: a quantized drafter

`qwen38-flash-next-w4a16b` — the drafter's stacked BF16 experts re-laid-out as per-expert
W4A16_NVFP4. **3.37 GiB saved** (122.83 → 119.46 GiB), **c=1 decode 37.47 mean** against a
36.45 ± 1.04 reference, so **no measurable cost**. Build takes 12 s: 205 of 206 shards hardlinked,
one rewritten, 512 experts → 4,608 tensors.

The drafter's 8.98% weight error costs nothing because **the target verifies every drafted token** —
drafter error surfaces as acceptance rate, never as wrong output. That makes the drafter the safest
place in this model to quantize hard.

**And it disproved the premise directly:** with the drafter quantized, the two workers select
*different* backends —

```
Worker           -> 'MARLIN'              (drafter, W4A16)
PleOffloadWorker -> 'FLASHINFER_CUTLASS'  (body, W4A4)
```

**vLLM selects the MoE backend per layer group, not globally.**

## What it took: four causes of one error, three of them ours

The failure was always `Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale'`, and it
had four independent causes:

1. **Wrong config key** — the runtime remaps the MTP module past the body's 48 layers
   (`mtp_start_layer_idx`), so it asks for `mtp.layers.48`, not `mtp.layers.0`. The shipped
   `mtp.*` wildcard matched any index; an exact key matched none.
2. **Wrong file** — the checkpoint carries **two** quantization configs. `config.json`'s embedded
   `quantization_config` is what the runtime reads; `hf_quant_config.json`, which we were editing,
   it does not.
3. **Half-fixed exclusions** — `exclude_modules` exists but is *empty* while `ignore` holds the real
   `mtp.*` entries. `key = 'exclude_modules' if 'exclude_modules' in q else 'ignore'` picks the empty
   one and reports a satisfying `0 -> 0`. **Strip both keys unconditionally.**
4. **Split gate/up `weight_scale_2`** — `process_weights_after_loading` takes `w13_weight_scale_2[:, 0]`
   for *both* halves and only **warns** on a mismatch. Separate scales are silently discarded, giving
   ~29% error on every up-projection weight in a server that starts and serves normally. Caught by
   reading the code, not by any benchmark.

Three of those were diagnosed by *reading* and each cost a ~12-minute server start to disprove. One
probe on `get_quant_method`, run **to completion**, settled it. **When two config edits fail
identically, stop editing and instrument.**

## Traps worth carrying elsewhere

- **A gate must ask the consumer's question, not the builder's.** Ours passed three times while the
  server failed: it read tensors as the *building* user, resolved the *checkpoint's* layer index, and
  loaded the file the runtime never consults. Resolve the remapped name, read as the serving uid,
  parse the file the runtime parses.
- **Editing a hardlinked file corrupts the source.** The build hardlinks every small file,
  `config.json` included, so editing the derived checkpoint rewrote the production one — same inode,
  `links=2`. For a while prod declared BF16 drafter experts as W4A16. **`stat` the link count before
  writing into a derived checkpoint**, or copy rather than link editable files.
- **`safe_open` reports permission-denied as `FileNotFoundError`.** A build run without `--uid`
  writes `0600 root:root` and the server dies 520 s later claiming the file does not exist.
- **Verify the quantizer by idempotence, holding the global scale fixed.** Re-quantizing an
  already-NVFP4 tensor must reproduce it exactly (we get rel L1 **0.000000%**, only the ±0 code alias
  differing). Letting `weight_scale_2` float changes the block grid and destroys representability —
  that produced a false "convention is wrong" verdict. The 9% error on raw BF16 weights is NVFP4's
  genuine cost, not a bug.
- The codebook `argmin` allocates 16× the tensor; `torch.bucketize` on the e2m1 midpoints is exact
  and reduces the whole build to seconds.

## Remaining backends

| backend | status |
| --- | --- |
| `FLASHINFER_CUTLASS` | what `auto` picks; every measurement here used it |
| `flashinfer_b12x` | selectable once the drafter is quantized — then **faults** (vllm#50189) |
| `triton`, `cutlass` | hit the SM120/121 CUTLASS SMEM overflow (99 KiB budget vs a 228 KiB assumption) |
| `TRTLLM`, `CUTEDSL`, `VLLM_CUTLASS`, `MARLIN`, `HUMMING` | untried, **no field evidence favours any on sm_121** |
