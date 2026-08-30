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
