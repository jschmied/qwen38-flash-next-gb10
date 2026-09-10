# Roadmap: a Local-Hessian NVFP4 rebuild of Flash-Next

Written 2026-09-09 from the day's measurements. **The goal is quality at zero inference cost** — same
scheme, same size, same speed, better calibration. Not a speed project.

## Why this is worth a roadmap at all

- **The calibration axis is real, not a no-op.** On `lm_head` of the 27B, NVIDIA's Local-Hessian scales
  differ from plain-max on **69.77 %** of groups and reconstruct the BF16 weights **1.0 pp better**
  (8.482 % vs 9.483 %, ~10.6 % relative). `headcmp`, 2026-09-09.
- **The search is not the cost.** ~6 min for the whole 125 B model. `lhbench`.
- **Nobody has published one.** NVIDIA's Flash-Next build is **MSE-calibrated** (their README says so).
  Our RadixArk build is 0.46.0, also not Local-Hessian. So this is new work, not a re-derivation.
- **The prize is the experts.** The head is a 1.27 B proxy; the 126 GB lives in 48 × 512 routed experts,
  and NVFP4 weights sit 4.5× further from BF16 than FP8 (12.1 % vs 2.7 %). Closing part of that costs
  nothing at inference.

## Stage gates — stop at the first one that fails

| # | gate | cost | status |
| --- | --- | --- | --- |
| 0 | **Does LH's weight-space win survive in OUTPUT space?** | ~20 min | **PASSED 2026-09-09 21:2x — 3.067 % vs 4.370 %, LH better by 1.303 pp (29.8 % relative) on 11,905 real activation vectors, vs 1.001 pp (10.6 %) in weight space. The margin GROWS in output space.** |
| 1 | **Expert routing coverage** (reframed: engagement was settled by the 0.46 changelog + the Triton fast path). Does our corpus route enough tokens to EVERY expert for a per-expert Hessian? | ~20 min | **PASSED 2026-09-09 22:17 — 0 of 512 experts below 10,000 assignments; skew only 3.7×. Per (layer, expert) the worst gets ~1,635 on a 33-prompt corpus and ~42,800 scaled to a real calibration.** |
| 2 | **Does one real expert layer behave like the head?** | ~1 h | **PASSED 2026-09-09, CORRECTED 2026-09-10 00:3x — LH better by median 22.2 % (range 20.2–24.1 %) across 32 experts with ≥300 genuinely routed rows, against the head's 29.8 %. The first pass reported 24.3 %/53.5 % and was contaminated: half the captured rows were batch padding, which routes to experts 0–9 because a zero row softmaxes to uniform.** |
| 3 | **The build** | see below | gated on 0–2 |
| 4 | **Validation before adoption** | logprob divergence vs BF16, then agent-task quality | gated on 3 |

## Stage 3, costed

Layer-by-layer, so the source never lands whole:

| | |
| --- | --- |
**Staging decision (user, 2026-09-09): fetch BF16 slices to the PBS server, not to the GB10.**
`10.0.0.70:/mnt/bulk/hf/` has **1.76 TB free** against the GB10's 293 GB, so the whole 335.3 GiB source
lands there once and is **retained** — a second calibration variant then costs zero download instead of
another 9.3 h. The GB10 pulls one layer at a time over LAN (PBS reads ~97 MB/s, disk-bound on its
two-disk mirror, so ~1 h of I/O spread across the whole run) and never holds more than a layer plus the
growing output. `hfget.sh` already writes `SOURCE.json` with per-file publisher hashes there, and
`hfverify.py` now checks both sha256 and git-blob-sha1, so the staged source is verifiable at 19/19
rather than 4/19.

| BF16 source fetch | 335.3 GiB at our measured **~10 MB/s** = **9.3 h**, unattended, free. Byte-range per layer (`hfpull_tensor.py`), so peak disk is one layer + the growing output ≈ **133 GiB** of our 294 GB |
| calibration | forward passes (**unmeasured** — the real cost) + ~6 min of search |
| output | born locally: **no return leg at all** |
| resumability | `checkpoint_dir` + `_CheckpointState` persist per-layer progress; a killed run resumes at its layer |
| memory | one decoder layer (~7 GiB) + activation cache (~43 GB at 2,048 samples) — fits 121 GiB |

Renting is a **scheduling** decision, not a capability one: the GB10 is the machine we work on, and a
multi-day run monopolises it. If stage 2 says the forward passes are slow here, rent; otherwise local.

## Decisions — SETTLED 2026-09-09 by the user

**1. ModelOpt: 0.47.0.dev80 or newer. DONE.** Installed from GitHub main into
`/opt/llm/modelopt-lib-047`: **0.47.0rc1.dev31+g2d356434**, commit dated 2026-09-09 16:35 — against
NVIDIA's `g913f5e224` (2026-08-20) for their 27B, so 20 days newer. 0.46.0 is the newest on PyPI; this
had to come from git. The serving venvs are untouched (`pip --target`, pure-Python wheel).

**2. Scope is ours to choose — the "keep it identical to RadixArk" constraint is dropped.**
The user's objection is correct and the earlier advice was advice we have never followed: prod already
serves `qwen38-flash-next-fp8head`, i.e. RadixArk's build with **Unsloth's FP8 head swapped in**, because
the RadixArk NVFP4 head measured **2.4 % worse NLL, 8/8** (`prod-dflash2-unsloth-head`, 2026-08-25).
So we have already changed scope, deliberately, on measured evidence.

The confounding worry was real but misplaced: it applies to *attribution*, not to *deployment*. Build
what we want to serve, and keep the current prod checkpoint as the A/B control — "new build vs current
prod" answers the deployment question directly. Attribution to the calibration axis is what gates 0–2
are for; they isolate it **before** the build, so the build does not have to.

Concrete consequence: since we are already replacing the head, the sharpest scope question is whether an
**LH-calibrated NVFP4 head can match the FP8 head we ship** — which is the original 2.4 % NLL question,
and `headcap`'s activations answer it with no build at all.

**3. Calibration data: agent trajectories + prose. And it IS required.**
There is no data-free Local-Hessian: the objective is `dw · H · dwᵀ` with **H = XᵀX** accumulated from
input activations, so it needs forward passes over real text. `layerwise_calibrate` raises
`forward_loop must not be None`, and `_register_local_hessian_input_hooks` exists precisely to capture
those activations. Data-free weight-only calibration is exactly the **max/MSE baseline that LH beats** —
asking for LH without data would silently get us the baseline.

Material we already hold, no download needed:
- `runners/dv/dv_prompts.json` (601 KB of real agent prompts)
- the 49,902-token agent transcript with 47 tools from the `tcorrupt` repro
- `runners/vpp/p5960.txt`, `p_prose.txt`, `p1999.txt` (prose)

This makes the build **ours, tuned to our traffic** — not a replication of NVIDIA's Nemotron-calibrated
recipe. State that whenever the checkpoint is described.


## Sharding policy: one component per shard (decided 2026-09-09)

**Why**: variants should cost only what changed. Measured on the two head variants we already hold,
by **inode** (not size): `qwen38-flash-next-nvfp4` and `-fp8head` share **202 of 206 files, 111.0 GiB
of literally the same bytes**; only 4 files, 11.83 GiB, differ. A head swap should cost the head, and
it currently costs 11.83 GiB because `lm_head` shares shard `model-bf16-00012` with `embed_tokens`.

**Neither exporter does this, so it is a post-process.** `layerwise_export.py` shards by *decoder
layer* (`model-layer-NNNNN.safetensors` + `model-tail.safetensors`); `unified_export_hf_streaming.py`
shards by *size* (`max_shard_size`). On Flash-Next the PLE is named
`model.language_model.layers.N.ple.*`, i.e. **inside** the layers — so a per-layer export would bundle
the 49.2 GiB PLE with the experts and every calibration variant would re-ship all of it.

**The invariant: a shard holds tensors from exactly ONE component.** Splitting a component across
several shards is fine and expected (73.3 GiB of experts cannot be one file); putting two components in
one shard is not, because then changing either one re-ships both. There is no catch-all "tail" bucket —
that is two or more components in one shard by another name, and it is what the current build does
(`everything else`: 1 file, 3.4 GiB) and what makes a head swap cost 11.83 GiB instead of 0.7.

| component | name rule | size | may span shards | changes per calibration variant? |
| --- | --- | --- | --- | --- |
| routed experts | `*.mlp.experts.*` | 73.3 GiB | yes — per layer is the natural cut (48) | **yes** — the only part that does |
| PLE n-gram | `*.ple.*` | 49.2 GiB | yes | no |
| `lm_head` | `lm_head.*` | ~0.7 GiB NVFP4 | no | only if the head is the variable |
| `embed_tokens` | `*.embed_tokens.*` | ~1.3 GiB | no | no — **must not share with `lm_head`** |
| self-attention proj | `*.self_attn.*` | | yes | no (unless scope changes) |
| GDN / linear-attn proj | `*.linear_attn.*` | | yes | no (unless scope changes) |
| MTP | `mtp.*` | | yes | no |
| hyper-connection | `*hyper_connection*` | small | no | no |
| norms | `*norm*` not matched above | small | no | no |
| router gates | `*.mlp.gate*`, `*shared_expert*` | small | no | no |

Small components still get their own shard even at a few MB — the file count goes up (roughly 40–90 vs
today's 206, depending on the expert cut) and that is the intended trade: a component is swappable iff
nothing else rides in its shards.

**Consequences.** Publishing: base once, then each variant is a branch costing only its component's
shards (HF reuses LFS objects across revisions; `huggingface_hub` skips files whose hash already
exists). Locally: hardlink every unchanged shard between variants — we already get 111 GiB of that by
accident, and this makes it exact. Fetching: `hfpull_tensor.py` gets whole files instead of byte ranges.

**Verification the re-shard must pass** (it rewrites a 126 GiB checkpoint, so it is not allowed to be
"probably fine"): every tensor present exactly once across the new index; each tensor's payload bytes
**sha256-identical** to the source; the rewritten `model.safetensors.index.json` resolves every key the
model asks for; and the reshaped checkpoint serves. Sizes are not evidence — `verify-checksums-not-sizes`.


### Sharding plan, validated against the real checkpoint (2026-09-10)

`/opt/llm/runners/reshard.py --plan` classified **all 296,475 tensors** of
`qwen38-flash-next-nvfp4` — **zero unclassified**, so the component rules partition the checkpoint.
Plan at an 8 GiB shard cap:

| component | tensors | GiB | shards |
| --- | ---: | ---: | ---: |
| **experts** | 294,914 | **68.0** | 9 |
| ple | 138 | 47.7 | 7 |
| linear_attn | 324 | 3.9 | 1 |
| hyper_connection | 387 | 1.2 | 1 |
| **lm_head** | 1 | **1.2** | 1 |
| embed_tokens | 1 | 1.2 | 1 |
| self_attn | 108 | 1.1 | 1 |
| visual | 333 | 0.8 | 1 |
| shared_expert | 192 | 0.4 | 1 |
| mtp | 29 | 0.2 | 1 |
| router_gate | 48 | 0.1 | 1 |
| **TOTAL** | 296,475 | 125.9 | **25** |

**What it buys, measured rather than estimated:**
- a **calibration variant costs 68.0 GiB** (the experts alone), against 125.9 for a whole checkpoint;
- a **head variant costs 1.2 GiB**, against the **11.83 GiB** the current layout forces because `lm_head`
  shares a shard with `embed_tokens` (measured by inode across our two existing head variants);
- 25 shards, not the 40–90 I guessed — the earlier estimate was pessimistic.

**One layout fact for the build:** this NVFP4 checkpoint stores experts **individually** (294,914 tensors,
e.g. `layers.N.mlp.experts.J.down_proj.*`), while the BF16 source stores them **fused**
(`experts.gate_up_proj [512,1280,2560]`). The re-shard runs on whatever layout the export produces; the
component rules match both, since they key on `.mlp.experts.` either way.

**`--execute` is written and verified (2026-09-10 02:2x).** It reads each tensor's payload by byte range
from the source shard, writes component-grouped shards with a rebuilt 8-byte-aligned header, then
**re-reads every tensor from the NEW file and compares sha256 against the source bytes** — sizes are not
evidence. A `--only <component>` flag runs it on one component so correctness can be proved without
spending 126 GiB.

Verification run on the `mtp` component of the live checkpoint: **29 tensors, 172.7 MiB, 29/29
sha256-verified, 0 mismatches**, and the written shard loads as valid safetensors with correct shapes,
dtypes and all-finite values. Tool at `/opt/llm/runners/reshard.py`.

Still to do before a full run: the index rewrite is only emitted when `--only` is absent, and a whole-
checkpoint pass needs ~126 GiB of free space alongside the source (we have 293 GB, so it fits, but there
is no reason to spend it until there is a rebuilt checkpoint to shard).

## The trap that would waste the whole run

`cfg = NVFP4_DEFAULT_CFG; cfg["algorithm"] = {"method": "local_hessian"}` calibrates **zero modules** and
prints `Calibration complete.` — that config has dynamic block scales, so a per-block search has nothing
to optimise, and there is no warning. Use **`mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`** and verify by the
**iteration count** (`MSE weight calibration: N it`), never by the completion line.

## The cheap alternative, checked first every time

Someone may publish a Local-Hessian Flash-Next build. The HF tree API gives per-file `lfs.oid` and
`hf_quant_config.json` without downloading anything, so checking the field costs one HTTP call and has
already invalidated two of our plans this month.
