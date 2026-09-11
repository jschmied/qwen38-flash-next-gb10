POSTED 2026-09-11 09:33 → https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4/discussions/13
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

**The causal test.** We rebuilt at *your* granularity (`--scale2-block 128`, plain max, nothing else
changed) and scored it on the same passages. The advantage collapses:

| vs your checkpoint | overall | excluding Devanagari (n=53) | Devanagari (n=6) |
| --- | --- | --- | --- |
| per-expert | **−0.0152**, t = −3.92 | −0.0069, t = −2.99 | −0.0891, t = −19.08 |
| your granularity, rebuilt | −0.0012, t = −0.72 | **+0.0002, t = +0.12** | −0.0133, t = −2.44 |

At block granularity the general gain is zero and 15 % of the Devanagari gain survives. On a separate
15-passage set a **plain-max** per-expert build — no calibration data at all — captured 69 % of the
overall gain and 87 % of the Devanagari gain, and beat Hessian calibration's increment only by 0.0054
(t = −1.18, not a result).

So the actionable part is one line in your export: **derive `weight_scale_2` per expert.** The
calibration method is not what matters here.

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
  captures most of the gain. The mechanism is granularity, not the Hessian.
- Not that 0.0152 matters in practice. It is ~0.9 % relative NLL and we have not run a task eval.
- Not that this generalises past Qwen3.8-Flash-Next on GB10 at TP1.

## Method, if they want to reproduce

Build driver, verifier and the combining-mark probe are in
[jschmied/qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10); the merge is
index-only and writes zero bytes. Held-out set and per-passage numbers in `notes/data/`.

---

**Posted** as discussion #13 after the user's explicit go (2026-09-11 09:32). Every figure was
re-verified against `/opt/llm/runners/results/nll.jsonl` immediately before sending: 59 passages,
70,734 tokens, 11 groups, and all eight paired statistics reproduce exactly.
