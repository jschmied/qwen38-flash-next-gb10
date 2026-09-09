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
| 0 | **Does LH's weight-space win survive in OUTPUT space?** `‖X(W−Ŵ)ᵀ‖` on real activations, LH vs max scales, same head | ~20 min | **`headcap` running** (queued behind `dv2`) |
| 1 | **Does the fused-MoE path actually engage Local-Hessian?** `_register_local_hessian_input_hooks` has an expert path keyed on `_current_expert_idx`; weights it cannot pair fall back to **plain MSE with a warning**. On 512 experts that is the difference between doing the experiment and thinking we did | ~30 min, 1 layer | not started |
| 2 | **Does one real expert layer behave like the head?** Quantise layer 24 LH vs MSE, output error on real activations | ~1 h | not started |
| 3 | **The build** | see below | gated on 0–2 |
| 4 | **Validation before adoption** | logprob divergence vs BF16, then agent-task quality | gated on 3 |

## Stage 3, costed

Layer-by-layer, so the source never lands whole:

| | |
| --- | --- |
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

## The trap that would waste the whole run

`cfg = NVFP4_DEFAULT_CFG; cfg["algorithm"] = {"method": "local_hessian"}` calibrates **zero modules** and
prints `Calibration complete.` — that config has dynamic block scales, so a per-block search has nothing
to optimise, and there is no warning. Use **`mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`** and verify by the
**iteration count** (`MSE weight calibration: N it`), never by the completion line.

## The cheap alternative, checked first every time

Someone may publish a Local-Hessian Flash-Next build. The HF tree API gives per-file `lfs.oid` and
`hf_quant_config.json` without downloading anything, so checking the field costs one HTTP call and has
already invalidated two of our plans this month.
