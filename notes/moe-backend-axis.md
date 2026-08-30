# The MoE kernel axis is closed while MTP is on

## Why we looked

Per-token bytes on `fp8head`, derived from the shard headers:

| group | per-token GiB |
| --- | --- |
| GDN (linear attn) | 1.95 |
| MoE experts (10 of 512) | 1.24 |
| hyper-connections | 1.19 |
| QSA (full attn) | 0.59 |
| lm_head | 0.59 |
| shared_expert | 0.44 |

At c=1 the experts are **20%** of the budget — but the dense side is read once per *step* and
shared across the batch, while each sequence routes to its own ten experts. The share crosses
50% at **c=4** and reaches **80% at c=16**. Every headline number we have published is c=1,
which is the regime where the experts matter least.

So: a better FP4 MoE kernel is the largest untouched lever *at load*.

## Three blockers, all real

⚠️ **"Global" is wrong — see the 2026-08-30 correction at the end of this file.**
`SpeculativeConfig.moe_backend` sets the drafter's backend independently.

**1. `--moe-backend` is global; this checkpoint is not.** `config.json`'s `ignore` contains
`mtp.*`, so `mtp.layers.0.mlp.experts.{gate_up,down}_proj` carry no scales (2 tensors, 4.86 GiB
BF16) while the body's 294,912 expert tensors are all scaled. One unquantized MoE layer in the
drafter vetoes the kernel choice for all 48 quantized layers:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
Expected one of ['triton','batched_triton','flashinfer_trtllm','flashinfer_cutlass','aiter'].
```

`map_unquantized_backend` (`fused_moe/oracle/unquantized.py:166`) raises rather than falling
back to a supported backend for the layers that cannot honour the request.

**2. It fails 10.5 minutes in.** Validation happens after weight load and torch.compile, not at
config parse. 17:51:56 engine init → 18:02:30 raise.

**3. With MTP off, b12x faults outright.** Illegal memory access during `profile_run`. It
surfaces at `hc.py:249 _hc_combine` inside Triton's `_init_handles`/`load_binary` — a kernel
*load*, not an execution, which is the signature of a context already poisoned by an earlier
async fault. `_hc_combine` is merely the next kernel to be JIT-loaded; the culprit is the b12x
MoE that ran before it.

## What this leaves

With MTP enabled the selectable set is `triton`, `batched_triton`, `flashinfer_trtllm`,
`flashinfer_cutlass`. `auto` already picks `FLASHINFER_CUTLASS`; `trtllm` is the known
silent-garbage path on sm_121 and `triton` is slower. **There is no move to make here.**

`CUTE_DSL_ARCH=sm_121a` was missing from `serve-fnext.sh` and has been added — it was required
for the b12x path on the 27B and had silently fallen off when the Flash-Next launcher was
written. It is harmless on the cutlass path.

## The lever this actually points at

Quantizing the **drafter's** MoE (`mtp.layers.0.mlp.experts.*`, 4.86 GiB BF16) would both free
~3.6 GiB resident and unlock the full backend set for the body. That is the prerequisite for
ever testing a native FP4 MoE kernel here, and it is a checkpoint edit of the kind we have done
four times already.

Not yet attempted.

## 2026-08-30 — de-risking the drafter-MoE build, before touching any weights

Four things established without writing a byte of new checkpoint.

**1. The config edit works, and touches nothing else.** Offline gate (~10 s, no GPU) via
`ModelOptMixedPrecisionConfig.from_config` + `_resolve_quant_algo`:

| probe | as shipped | after removing `mtp.*` from `exclude_modules` + adding a `quantized_layers` entry |
|---|---|---|
| `mtp.layers.0.mlp.experts` | **None** (unquantized) | **NVFP4** |
| `model.language_model.layers.0.mlp.experts` | NVFP4 | NVFP4 |
| `lm_head` | FP8_PB_WO | FP8_PB_WO |

**2. The drafter's quant config is resolved independently.** `get_draft_quant_config` reads the
*draft* model config (`model_executor/models/utils.py:883`), and `mtp.py:115-141` injects it. Since
our speculative config points at the same directory, one edit reaches both.

**3. Only one shard changes.** Both drafter expert tensors live in `model-bf16-00011.safetensors`
(8.08 GiB of 206 shards), so the build can hardlink 205 shards and rewrite one — which matters,
because the box has **21 GB free**.

**4. The layouts differ, and that is the work.** The body stores experts *per-expert, per-projection*
(`experts.{N}.{down,gate,up}_proj.weight` U8 + `weight_scale` + `weight_scale_2` + `input_scale`,
**6,144 tensors for layer 0 alone**). The drafter stores the *fused, stacked* form —
`gate_up_proj [512, 1280, 2560]`, `down_proj [512, 2560, 640]`, two BF16 tensors — which is what
`FusedMoE` holds internally as `w13`/`w2` (`mtp.py:358` maps
`"gate_up_proj": ["gate_proj", "up_proj"]`). The split is clean (gate = first 640 rows, up = last
640), so a per-expert emission is mechanical, if bulky.

### The fork that has to be decided before building

| | W4A16_NVFP4 (weight-only) | NVFP4 (W4A4) |
|---|---|---|
| `input_scale` | **not needed** | required |
| calibration | **none — data-free** | needs a calibration pass |
| memory | ~3.6 GiB freed | ~3.6 GiB freed |
| MoE backend | *"NVFP4 routed experts run via **Marlin W4A16**"* (`modelopt.py:2291`) | the scheme the CUTLASS/b12x kernels want |

So the cheap, data-free build most likely pins the MoE to **Marlin**, which is not the backend we
were trying to unlock — while the build that unlocks b12x is the one needing calibration, and
carries the trap that dropping `input_scale` while the algo stays `NVFP4` gives an uninitialised
activation scale and **zero characters of output, silently**.

**Status: the memory win is cheap and safe; the backend win is not the same build.** Unblocked and
ready to build either way, but the choice is a real one and worth making deliberately.

## 2026-08-30 — the W4A16 drafter checkpoint, built

`qwen38-flash-next-w4a16mtp`. Built in **12 s**: 205 shards hardlinked, one rewritten, 512 experts →
4,608 per-expert tensors, **3.37 GiB saved** (122.83 → 119.46 GiB).

### Getting the quantizer right, and one wrong verdict of my own

The NVFP4 convention was **derived from the checkpoint's own body tensors** rather than assumed:
low nibble first, 16-element blocks along the last dim,
`value = LUT[code] * block_scale(e4m3) * weight_scale_2`.

The validation took three attempts and produced one false alarm worth recording:

1. Forward-quantizing a BF16 drafter tensor gave **9.0% rel L1** — looked alarming.
2. An idempotence test (re-quantize an already-NVFP4-representable tensor) reported **2.54%** and I
   called the convention wrong. **The test was wrong.** It let `weight_scale_2` float, which changes
   the block-scale grid, so exact representability was destroyed by construction.
3. Holding `ws2` fixed: **rel L1 exactly 0.000000%**, scales bit-identical, and the only differing
   codes are the sign-magnitude ±0 alias, which dequantizes identically. Convention confirmed.

So the **9.0% is NVFP4's genuine intrinsic error on BF16 weights**, not a bug. And it matters less
here than anywhere else in the model: the drafter only *proposes* tokens and the target verifies
every one, so drafter weight error costs **acceptance rate, not correctness**. The drafter is the
safest place in this model to quantize hard, and the risk of this build is that it is slower, never
that it is wrong.

Encoder note: the codebook `argmin` allocates 16× the tensor. `torch.bucketize` on the e2m1
midpoints is exact and reduces the whole build to seconds.

### Two traps hit, both already in our own notes

- **`sudo -S` cannot coexist with any stdin redirect.** `printf pw | sudo -S patch -p1 < file` feeds
  the *patch file* to sudo as the password. Use `patch -i`, or write the script to a file.
- **`safe_open` reports permission-denied as `FileNotFoundError`.** The build ran under
  `systemd-run` *without* `--uid`, so it ran as root and `save_file` wrote `0600 root:root`; the
  server runs as `llm` and died after **520 s** with
  `No such file or directory: …/model-w4a16-mtp.safetensors` — on a file the gate had just opened
  successfully.

**The gate's real defect was narrower than the chown.** It opened every tensor **as the building
user**, so it passed. An offline gate exists to catch what a server start would catch, and it can
only do that if it reads under the identity the server actually runs as. Preflighting
`sudo -u llm head -c 8 <shard>` costs nothing and replaces a nine-minute failure.

### The config key must use the RUNTIME layer index, not the checkpoint's

First serve attempt died after 630 s with:

```
AttributeError: Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale'
  for checkpoint weight 'mtp.layers.48.mlp.experts.0.down_proj.weight_scale'
```

**`mtp.layers.48`, not `mtp.layers.0`.** The MTP module is remapped past the body's 48 layers by
`mtp_start_layer_idx` — the same remapping `mtp.py` applies to ignored layers through
`_remap_ignored_layers`. The shipped `exclude_modules` entry is the wildcard `mtp.*`, which matches
any index; the exact key `mtp.layers.0.mlp.experts` we added matches **none**. So
`_resolve_quant_algo` returned `None`, `FusedMoE` was built **unquantized** with no scale
parameters, and the loader then arrived with `weight_scale` tensors it had nowhere to put.

Fix is config-only — the 4,608 quantized tensors were correct throughout. Declare **both** indices
(or a wildcard):

```python
for i in (0, 48):
    ql[f'mtp.layers.{i}.mlp.experts'] = {'quant_algo': 'W4A16_NVFP4'}
```

**The gate failed the same way twice in one session, for the same reason.** It verified
`mtp.layers.0` because that is what the *checkpoint* calls the layer, and earlier it opened tensors
as the *building* user rather than the serving one. Both times it asked the question the builder
would ask instead of the question the consumer asks. A gate is only worth its runtime if it queries
the way the runtime queries: **resolve the remapped name, and read as the serving uid.**

One consolation: this failure mode is *loud*. An unquantized layer meeting quantized weights raises
immediately. The inverse — dropping `input_scale` while the algo stays `NVFP4` — produces zero
characters of output with no error at all, which is why the W4A4 path deserves more caution than
this one did.

### Why no config key works: the drafter never reaches the mixed-precision config

Instrumented `ModelOptMixedPrecisionConfig.get_quant_method` to print prefix, layer class and
resolved algo for every expert layer. One server start settles it:

```
96 lines, all:  prefix='language_model.model.layers.X.mlp.experts'  cls=RoutedExperts  algo='NVFP4'
 0 lines:       prefix='mtp.…'
```

⚠️ **This conclusion was wrong — see the correction below.**

**The drafter's expert layer never reaches that class at all.** So no `quantized_layers` key can
help — not `mtp.layers.0`, not `mtp.layers.48`, not a wildcard. The drafter is served by a
*separate* quant-config object built by `get_draft_quant_config`
(`model_executor/models/utils.py:883`) from the draft model config, and `mtp.py:115-141` mutates
only that object's `ignored_layers` / `exclude_modules` — **never its `quantized_layers`**.

Consequences, and they reshape the item:

- A drafter sharing the target's `hf_quant_config.json` can only be **excluded** or take whatever
  single algo that config path yields. Declaring a *different* scheme for it (our W4A16 plan) has
  no route through the shared file.
- So the two remaining options are: give the drafter **its own directory** with its own
  `hf_quant_config.json` pointed at by `speculative_config.model`, or quantize it to the **same**
  scheme the body uses (NVFP4 W4A4), which needs `input_scale` and therefore calibration.
- The 4,608-tensor checkpoint is **not wasted** — the weights are correct and verified, and they are
  the right weights for the separate-directory route. What is wrong is only where the declaration
  lives.

Incidental find while instrumenting: `modelopt.py` carried a **leftover debug `print`** in
`get_quant_method` from our earlier lm_head work, firing on every `lm_head` construction, plus two
undocumented backups (`.prepbwo`, `.pre-pcpt`). Removed. Together with the `mtp.py` patch and the
PLE backport, the serving venv holds at least four local modifications that no reinstall preserves —
worth an inventory rather than discovering them one at a time mid-debug.

### Correction: the probe was terminated before the drafter was built

The "zero `mtp.*` lines" reading above does **not** support its conclusion, and the conclusion is
withdrawn.

**Two independent problems with it.**

1. **The probe run was stopped too early.** Its wait loop broke as soon as *body* expert lines
   appeared, slept 20 s, and killed the server. The MTP drafter is constructed **after** the main
   model, so the absence of `mtp.*` lines is most likely an artifact of stopping first — absence of
   evidence produced by ending the observation early, which is the same mistake as the two-request
   prefix-cache probe.
2. **The mechanism it proposed does not exist.** `config/speculative.py:798-810` shows that for
   `method == "mtp"` the draft config *does* inherit the target's quantization:

   ```python
   if self.method == "mtp":
       ...
       if not self.quantization:
           self.quantization = self.target_model_config.quantization
   ```

   and that value is passed to the draft `ModelConfig` at line 955. So `get_draft_quant_config`
   should return a mixed-precision config, not `None`, and the claim that the drafter is "served by a
   separate config object that never sees `quantized_layers`" has no support.

**Consequence for the fix under test:** the `draft_quant_config is None` fallback patched into
`mtp.py` is then a **no-op** for this configuration, and the run should fail exactly as before. That
is a real prediction and the running A/B will settle it. The `w2_weight_scale` failure is genuine and
reproducible; **its cause is once again unknown.**

What still stands from that investigation: the config *does* resolve correctly offline for both the
checkpoint name and the remapped runtime name, and `W4A16_NVFP4` *does* have a `RoutedExperts`
method (`modelopt.py:2565`). Whatever is wrong sits between those two facts.

### Root cause: the checkpoint carries TWO quantization configs

The `w2_weight_scale` failure was never a vLLM bug. This checkpoint has **two** quantization
declarations, and the runtime reads the one we were not editing:

| | `hf_quant_config.json` | `config.json → quantization_config` |
|---|---|---|
| what we edited | ✅ | ✗ |
| **what the runtime reads** | ✗ | ✅ |
| `mtp` in `quantized_layers` | present | **absent** |
| `mtp.*` in exclusions | removed | **still there** |

So the drafter's experts were simultaneously *excluded* and *undeclared*: `_resolve_quant_algo`
returned `None`, `FusedMoE` was built unquantized, and the loader then arrived with `weight_scale`
tensors that had no home.

Confirmed with a probe on `get_quant_method` that ran **to completion** rather than being stopped
early:

```
96 x  prefix='language_model.model.layers.X.mlp.experts'  cls=RoutedExperts  algo='NVFP4'
 1 x  prefix='mtp.layers.48.mlp.experts'                  cls=RoutedExperts  algo=None      <-- here
```

The drafter *does* reach the mixed-precision config, and the layer *is* a `RoutedExperts` — both of
my earlier hypotheses were wrong, and the probe settled it in one run.

**Two lessons, both about the gate rather than the model.**

1. **Gate the file the runtime reads.** Our offline gate loaded `hf_quant_config.json` and passed
   perfectly against a file nothing consults. That is the third variant of the same mistake in one
   session — after reading as the *building* user instead of the serving one, and resolving the
   *checkpoint's* layer index instead of the runtime's remapped one. The gate kept asking the
   builder's question.
2. **`exclude_modules` can exist and be empty while `ignore` holds the real list.** A first fix that
   did `key = 'exclude_modules' if 'exclude_modules' in q else 'ignore'` picked the empty list and
   reported a satisfying `0 -> 0` while `mtp.*` sat untouched under `ignore`. **Strip both keys
   unconditionally.** This was only caught because `_resolve_quant_algo` does not consult exclusions,
   so a passing resolve could not have proven them clear — checking anyway is what surfaced it.

### ⚠️ Editing a hardlinked file corrupted the production checkpoint

The build hardlinks 205 of 206 shards **and every small file, including `config.json`**. Editing
`w4a16mtp/config.json` in place therefore rewrote **`fp8head/config.json` — the same inode**:

```
25169287  links=2  qwen38-flash-next-fp8head/config.json
25169287  links=2  qwen38-flash-next-w4a16mtp/config.json
```

For a while the production checkpoint declared `mtp.layers.{0,48}.mlp.experts` as `W4A16_NVFP4`
while its actual `mtp` weights are BF16 stacked — it would have failed to start with the same
missing-scale error, and the cause would have looked like corruption of a checkpoint nobody had
touched.

Restored by copying to break the link, then reverting exactly the two edits (drop the mtp
`quantized_layers` keys, restore `mtp.*` and `model.mtp.*` to `ignore`). Verified: separate inodes,
`links=1` each, `fp8head` clean.

**Rules this earns:**

- **A hardlink-based build makes every small file shared.** Cheap for shards; a trap for anything
  editable. Either copy the config files instead of linking them, or `stat` the link count and break
  it before the first write.
- **Check `st_nlink` before editing anything inside a derived checkpoint.** One `stat` would have
  caught this before the write rather than after.
- The build now runs **as the `llm` user** from `/opt/llm/build/`, which fixes the earlier
  `0600 root:root` problem at the source rather than with a follow-up `chown`. Note the scratchpad
  is `drwx------ jschmied`, so an `llm`-owned build can neither read scripts nor write logs there —
  the first attempt failed instantly with `Keine Berechtigung`.

## 2026-08-30 — the quantized drafter works: 3.37 GiB free, no measurable cost

`qwen38-flash-next-w4a16b` serves.

| | result |
|---|---|
| loads and serves | **yes**, 790 s |
| c=1 decode | **38.0 / 36.8 / 37.6**, mean **37.47** tok/s |
| reference (BF16 drafter) | 36.45 ± 1.04 |
| TTFT | 1.72 / 1.61 / 1.77 s |
| memory saved | **3.37 GiB** |
| gate/up scale warning | none |

**The drafter's 8.98% weight error costs nothing measurable at c=1** — mean is *above* the
reference, comfortably inside the 6.9% noise floor. That matches the argument made before building:
the target verifies every drafted token, so drafter weight error surfaces as acceptance rate, and
here not enough of it to move single-stream decode.

### vLLM selects the MoE backend PER LAYER GROUP, not globally

```
Worker           -> 'MARLIN'              NvFp4 MoE backend    <- the drafter (W4A16)
PleOffloadWorker -> 'FLASHINFER_CUTLASS'  NvFp4 MoE backend    <- the body (W4A4)
```

Two different backends in one server. So the premise this whole item rested on — *"one unquantized
MoE layer in the drafter vetoes the kernel choice for all 48 quantized layers"* — is only true while
the drafter is **unquantized**. Once it is quantized it simply takes its own backend, and
`modelopt.py:2291` routes W4A16 NVFP4 routed experts to Marlin exactly as predicted. vLLM says so
plainly: *"Your GPU does not have native support for FP4 computation … Weight-only FP4 compression
will be used leveraging the Marlin kernel."*

At c=1 that costs nothing measurable — the drafter is small enough that its kernel choice does not
move single-stream decode.

### What it took: four causes, three of them mine

The `w2_weight_scale` failure had **four** distinct causes, eliminated one at a time, and only one
was a vLLM behaviour:

1. **Wrong config key** — `mtp.layers.0` where the runtime remaps to `mtp.layers.48`.
2. **Wrong file** — the runtime reads `config.json`'s embedded `quantization_config`, not
   `hf_quant_config.json`. The gate passed against a file nothing consults.
3. **Half-fixed exclusions** — `exclude_modules` existed but was empty while `ignore` held the real
   `mtp.*` entries; an `if key in q else` picked the empty one and reported a satisfying `0 -> 0`.
4. **Split gate/up `weight_scale_2`** — vLLM only *warns* and takes `[:, 0]` for both halves, so
   separate scales are silently discarded. Caught by reading `process_weights_after_loading`, not by
   a benchmark; it would have served happily with ~29% error on every up-projection weight.

Plus one self-inflicted incident along the way: editing the hardlinked `config.json` **corrupted the
production checkpoint** (see above).

**The lesson that would have saved the most time:** three of those four were diagnosed by *reading*
and each cost a ~12-minute server start to disprove. The two that actually resolved things came from
**instrumenting** `get_quant_method` and from **reading the consumer's code path**. When two config
edits fail identically, stop editing and instrument.

## The b12x axis, closed for good — and for a different reason than before

With the drafter quantized, `--moe-backend flashinfer_b12x` was run explicitly:

| | before (BF16 drafter) | now (W4A16 drafter) |
|---|---|---|
| `not supported for unquantized MoE` | raised, 10.5 min in | **0 occurrences** |
| backend selected | never reached | **`FLASHINFER_B12X`**, both workers |
| outcome | config-time rejection | **`Triton Error [CUDA]: illegal memory access`**, 730 s |

So the unlock worked exactly as intended — and the kernel behind it does not run on this hardware.

The fault surfaces at `_hc_combine` inside Triton's `_init_handles` / `load_binary`, i.e. during a
kernel **load**, which is the signature of a context already poisoned by an earlier async fault;
`_hc_combine` is simply the next kernel to be JIT-loaded. That is the **same signature** we recorded
for b12x with MTP off, so the b12x MoE path faults on sm_121 for this model regardless of the
drafter.

**Final state of the item, both halves answered:**

- **Memory win: real and free.** 3.37 GiB back, c=1 decode 37.47 mean against a 36.45 ± 1.04
  reference. Ship it if the space is wanted.
- **Backend win: unobtainable.** b12x is now *selectable* and *faults*. Nothing further to try here
  without an upstream kernel fix, and `auto` already picks `FLASHINFER_CUTLASS` for the body, which
  works.

The item entered the day ranked first on the belief that the drafter was blocking a faster kernel.
It was blocking it — and the kernel is broken anyway. Both facts had to be established separately,
and only the second one closes the axis.

## 2026-08-30 — correction: `--moe-backend` is NOT global

This file has claimed since August that *"`--moe-backend` is global; one unquantized MoE layer in the
drafter vetoes the kernel choice for all 48 quantized layers."* **The first half is false**, and it
is the reason this item sat at rank 0 for weeks.

`SpeculativeConfig.moe_backend` (`vllm/config/speculative.py:118`) sets the **drafter's** backend
independently. Its docstring names our exact situation:

> *"MoE backend to use for the draft model. When `None`, the draft model … drafter and generator
> require different MoE kernels (e.g. quantized generator with unquantized drafter)."*

So the supported configuration was always:

```
--moe-backend flashinfer_b12x --speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"flashinfer_cutlass"}'
```

No checkpoint rebuild, no re-quantisation, no 4,608 tensors. Found via **vllm#51960**, whose author
reports the identical trap — *"I spent a while concluding the configuration was impossible before
finding that field"* — and proposes adding one sentence to the error message, which is exactly what
would have saved both of us.

**What this costs and does not cost us.** The W4A16 drafter build is still worth 3.37 GiB at no
measurable decode cost, so it is not wasted — but it was the expensive route to a problem with a
documented one-line answer, and the error message we trusted (*"Expected one of [...]"*) reads as
*this combination is unsupported* when it is merely *this flag is the wrong one*.

### And our b12x fault is already reported upstream

**vllm#50189** (open since 2026-07-28): *"Xid 31 MMU fault (illegal write) with flashinfer_b12x MoE
backend under concurrent chunked prefill"*, SM120, `Qwen3.5-122B-A10B-NVFP4`. Their scoping is
precise and matches ours being a load-time/JIT-surfaced fault rather than an OOM: light traffic
clean, single prefills to 208k clean, vision clean — **only concurrent chunked prefill faults**, and
the same scenario passes 48/48 on the default non-b12x backend.

So the b12x MoE path has a known, unfixed memory-safety bug upstream. Ours is not a GB10-specific
misconfiguration.

### `SpeculativeConfig.moe_backend` is honoured on V1 and ignored on V2 — measured

Ran the documented configuration on the **unmodified** `fp8head` checkpoint:

```
--moe-backend flashinfer_b12x
--speculative-config '{"method":"mtp","num_speculative_tokens":2,"moe_backend":"flashinfer_cutlass"}'
```

The engine **accepted and logged it** —
`speculative_config: {'method': 'mtp', 'num_speculative_tokens': 2, 'moe_backend': 'flashinfer_cutlass'}`
— and then failed at 570 s with the original error:

```
ValueError: moe_backend='flashinfer_b12x' is not supported for unquantized MoE.
```

**Where it is read, and where it is not:**

| path | reads `spec_cfg.moe_backend`? |
|---|---|
| `v1/spec_decode/llm_base_proposer.py:1295-1300` | **yes** |
| `v1/worker/gpu/spec_decode/*` (V2 runner) | **no** — the string appears nowhere |
| `models/qwen3_8_flash_next/nvidia/mtp.py` | **no** |

So the field works for models routed through the V1 proposer and is **silently inert** for
Qwen3.8-Flash-Next, which runs the V2 model-runner MTP path — and #53896's newest commit
(`fb97542ccc`, today) is *"Limit Qwen3.8-Flash-Next to model runner V2"*, pinning this model to the
path that ignores it.

**Why this matters beyond us.** vllm#51960 proposes amending the unquantized-MoE error message to
point users at exactly this field. If that lands as written, users of any V2-runner model will be
directed to a config option that is accepted, echoed back in the engine config, and does nothing —
a worse failure than the current message, because it looks like it worked.

That is worth reporting, and it is a **measured** result rather than a code reading: the config was
verified present in the engine's own log before the run failed.
