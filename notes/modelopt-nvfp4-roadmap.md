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

## Decisions to make before starting

1. **ModelOpt version.** 0.46.0 is the newest on PyPI; NVIDIA's 27B used **0.47.0.dev80** from GitHub
   main. Match their toolchain (`pip install git+…Model-Optimizer`) or stay on the release and accept a
   different generation. Pin whatever we choose and record it — repo names lie, `hf_quant_config.json`
   does not.
2. **Scope: keep it identical to RadixArk.** 48 routed-expert layers, `*.self_attn.*` /
   `*.linear_attn.*` / `*.ple.*` / `lm_head` excluded. Changing scope *and* calibration in one build
   confounds the only question we are asking. Widening scope is a separate experiment.
3. **Calibration data.** NVIDIA used 2,048 samples of Nemotron post-training data. Ours would be our own
   agent traffic — arguably better for our deployment, but it makes the build **ours**, not a
   replication. Say which we are doing.

## The trap that would waste the whole run

`cfg = NVFP4_DEFAULT_CFG; cfg["algorithm"] = {"method": "local_hessian"}` calibrates **zero modules** and
prints `Calibration complete.` — that config has dynamic block scales, so a per-block search has nothing
to optimise, and there is no warning. Use **`mtq.NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`** and verify by the
**iteration count** (`MSE weight calibration: N it`), never by the completion line.

## The cheap alternative, checked first every time

Someone may publish a Local-Hessian Flash-Next build. The HF tree API gives per-file `lfs.oid` and
`hf_quant_config.json` without downloading anything, so checking the field costs one HTTP call and has
already invalidated two of our plans this month.
