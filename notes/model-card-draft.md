# DRAFT model card — not published, no HF token exists on this box

Written 2026-09-10 while the build is still pending, deliberately: the caveats below are written by
someone who remembers why each one matters. **Every `TBD` is a cell the build must fill; none may be
quietly dropped at publish time.** If a TBD cannot be filled, the claim it belongs to comes out.

---

# Qwen3.8-Flash-Next NVFP4 — Local-Hessian expert calibration

NVFP4 W4A4 on the routed experts of `Qwen/Qwen3.8-Flash-Next`, with the per-group weight scales chosen
by **Hessian-weighted search** (ModelOpt `local_hessian`, the method of arXiv 2608.28113 "H-Scale" from
the Qwen team) instead of the MSE/max sweep used by the published builds.

**Same format, same size, same speed as an ordinary NVFP4 build. Only the scales differ.** This is a
quality change with zero inference cost — not a speed or memory optimisation.

## What is actually different

| | this build | the usual NVFP4 build |
| --- | --- | --- |
| routed experts | NVFP4 W4A4, **Local-Hessian scales** | NVFP4 W4A4, MSE/max scales |
| everything else (PLE, attention, GDN, head, embeddings) | untouched, copied verbatim | — |
| calibration data | **our own agent traffic + prose** | NVIDIA's builds use Nemotron post-training data |
| toolchain | ModelOpt 0.47.0rc1.dev31+g2d356434 (GitHub main, 2026-09-09) | 0.46.x in the published builds |

## Evidence

Measured on one GB10 (DGX Spark, sm_121), before the build, to decide whether it was worth doing:

| | |
| --- | --- |
| **output error, `lm_head`** (NVIDIA's own published LH scales vs plain-max, 11,905 real activations) | **3.067 % vs 4.370 % — 29.8 % better** |
| **output error, real experts** (32 experts of one layer, ≥300 routed rows each) | **20.2–24.1 % better, median 22.2 %** |
| weight-space error, same head | 8.482 % vs 9.483 % — 10.6 % better |

The advantage is **~3× larger in output space than in weight space**, which is what a Hessian-weighted
objective predicts: plain L2 weight error understates it because real activations do not excite every
weight direction equally.

| end-to-end quality vs the MSE baseline | **TBD** |
| serving throughput / TTFT vs the MSE baseline | **TBD — expected null; this is not a speed change** |

## Honest limits

- **This is not a reproduction of NVIDIA's recipe.** Calibrated on our agent traffic, not Nemotron.
  It is tuned to our deployment, which is a different claim.
- **Some experts are not Hessian-calibrated.** Experts that receive no routed tokens from the
  calibration corpus fall back to weight-only max. On a small probe (8,192 rows, one layer), 21 of 512
  experts got no rows. **The published build must state the per-layer counts** — TBD.
- Gate evidence is one layer of 48, and activations came from the quantised body via the eager path.
- The `lm_head` number uses NVIDIA's *published* scales, so it validates the method, not our pipeline;
  the expert number uses our own ModelOpt run.

## Provenance

- base: `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`
- source verified **144 of 144 files** against publisher hashes (132 sha256 + 12 git-blob-sha1)
- config: `NVFP4_W4A4_WEIGHT_LOCAL_HESSIAN_CFG`, group size 16
- calibration corpus: TBD (size, token count, per-layer expert coverage)

## Layout

Shards are **one component each** — experts, PLE, `lm_head`, `embed_tokens`, attention, GDN, MTP, norms,
gates never share a shard. So a variant costs only what changed: a recalibration is the 68 GiB of
experts, a head swap is 1.2 GiB. Variants ship as branches of this repo, reusing the unchanged LFS
objects.

## Not claimed

- no speed gain
- no memory saving
- no claim about non-Latin scripts or long-context retrieval, neither of which we evaluated
