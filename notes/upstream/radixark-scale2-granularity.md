DRAFT — needs the user's go, AND needs the `blk` measurement before it is worth sending.
HuggingFace discussion, RadixArk/Qwen3.8-Flash-Next-NVFP4 (2026-09-11).

---

## What we would tell them

We rebuilt the routed experts of your NVFP4 checkpoint on a single DGX Spark (GB10, sm_121, TP1) and
measured the result against yours on held-out text. Two things came out that you may want.

### 1. A possible improvement: per-expert `weight_scale_2`

Your checkpoint derives one gate/up `weight_scale_2` per **block of 128 experts** — 4 per layer,
uniform across every layer we censused (0, 1, 12, 24, 36, 47). We derive **one per expert**.

Held-out NLL/token, 59 Wikipedia passages / 70,734 scored tokens across 11 languages, fetched at
offsets no build had seen:

| | NLL/token |
| --- | --- |
| our rebuild | **1.7227** |
| your checkpoint | 1.7379 |

−0.0152, paired **t = −3.92** over passages (40/59 favour the rebuild). It splits into a small general
gain (−0.0069, t = −2.99 excluding Devanagari) and a large Devanagari-specific one (**−0.0891**,
consistent across all six of its passages). Thai is second at −0.0249; Hebrew is the one small
regression (+0.0021).

**TBD — the causal test.** A build at *your* block granularity, identical in every other respect,
is measured as arm `blk`. If its gain collapses, granularity is the mechanism and this is worth your
time. If it survives, we do not know the cause and this section should be cut to the bare
observation. **Do not send before that number exists.**

### 2. Two things your checkpoint gets right that we got wrong

Worth saying plainly because we broke both and it cost a day:

- `input_scale` = `amax / (E2M1_MAX * E4M3_MAX)` = `amax/2688`. We wrote `amax/6` — 448× too large.
- gate and up must share one `weight_scale_2`. vLLM fuses them into `w13` and on a mismatch silently
  applies the **gate** scale to both halves ([vllm#54974](https://github.com/vllm-project/vllm/issues/54974)),
  so a separate up scale is discarded at runtime.

Your checkpoint has **0/24,576** gate/up mismatches and the correct `input_scale` convention
throughout — we used it as the reference that found both of our bugs. Our `down_proj` export is now
byte-identical to yours, which is how we know the rest of the pipeline agrees.

With those broken, our build corrupted Thai combining marks (duplicated U+0E48/U+0E49) in 25–50 % of
copy tasks while yours was clean in 72/72. Weight-space reconstruction did not see it — it was 0.004
pp from yours while both contracts were violated — because it reconstructs each matrix with its own
scale, which is not what the runtime does.

## What we are NOT claiming

- Not that Local-Hessian calibration is why. A **plain-max** build with no calibration data at all
  captures most of the gain over your checkpoint. Whatever the mechanism is, it is not the Hessian.
- Not that 0.0152 matters in practice. It is ~0.9 % relative NLL and we have not run a task eval.
- Not that this generalises past Qwen3.8-Flash-Next on GB10 at TP1.

## Method, if they want to reproduce

Build driver, verifier and the combining-mark probe are in
[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10); the merge is
index-only and writes zero bytes. Held-out set and per-passage numbers in `notes/data/`.

---

**Gate:** needs the `blk` result, then the user's explicit go. Posting log entry required in
`notes/upstream/README.md` after.
