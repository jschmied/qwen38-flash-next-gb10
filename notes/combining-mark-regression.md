# The Local-Hessian rebuild corrupts Thai combining marks — 2026-09-10

**The build is verified correct and measurably worse.** Every structural check passed; the quality
check failed. That is the whole point of having both.

## Result

| arm | starts | responses | corrupt | exact copies |
| --- | --- | --- | --- | --- |
| **stock** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) | 2 | 72 | **0** | **72 / 72** |
| **LH rebuild** (`/opt/llm/models/fnext-lh`) | 1 | 24 | **6** | 12 / 24 |

Identical server flags both arms (`FN_MAXLEN=8192 FN_SEQS=4 FN_UTIL=0.75 FN_MTP=3 FN_SPEC_NODROP=1`,
port 8092, `flashnext`), same prompts, temp 0, seed = rep index. The **only** differing cell is
`FN_MODEL`.

## The damage is structural, not a wording difference

```
stock:  ผู้ใช้ต้องเข้าสู่ระบบก่อนที่จะแก้ไขข้อมูลนี้ได้
ours:   ผู้ใช้ต้องเข้าสู่ระบบก่่นที่จะแก้ไขข้อมูลน้ีได้
```

Two distinct faults in one line: `ก่อน` loses its base character **อ** and gains a **doubled mai ek**
(`่่`), and `นี้` comes back as `น้ี` with vowel and tone mark **reordered**. That is exactly the
ordering damage MiaAI reported in their #27.

## Why this is not the model's known nondeterminism

`temp0-not-reproducible-under-load` records that Flash-Next diverges run to run even at c=1. Three
things rule that out here:

1. **Deterministic per prompt within the arm** — th-1 3/3 corrupt, th-3 3/3 corrupt, th-2 0/3.
2. **Stock is flawless across two starts** — 72/72 exact. Divergence would not spare it entirely.
3. **The exact-copy rate halves**, 72/72 → 12/24. That is a broad capability difference, not a
   coin-flip on one token.

## Scope

Thai only. Devanagari, Arabic harakat, Hebrew niqqud and emoji ZWJ were clean in both arms. The
Mn/Mc/Me census correctly declined to score the Devanagari rewordings as corruption — the
discrimination that was tested synthetically before the probe was trusted.

## What it means for MiaAI #27 / our #37

**It cuts against our own hypothesis.** Our issue #37 proposed that #27's corruption comes from the
shipped build quantizing the GDN/linear-attention path with a five-month-old ModelOpt. Here the
signature appears in a build whose `linear_attn` is **excluded from quantization** — the damage came
from quantizing **experts**. So expert quantization is *sufficient* to produce this signature, and
#37's mechanism is not necessary for it. That does not disprove #37, but it removes this instance as
support and weakens the case. Worth telling them; needs the user's go.

## Consequences taken

- **Upload killed mid-transfer.** `upload_folder` commits once at the end, so the commit never
  landed: **no weights were ever published.** The repo has only ever held the card and the verifier.
- The public card now carries a ⛔ banner with these numbers and the example above.
- SWE run A not started — the canary failed first, and at n=10 it could not have detected this
  anyway (SE ~15 points against a defect the canary caught in 35 minutes).

## Open: the cause

Not identified. Candidates, cheapest first:

1. **The 33 no-rows experts** on plain-max — but that is the standard method every published build
   uses, so it should not be worse than stock.
2. **The 58 thin experts** (1–63 routed rows) — under-determined Hessians could be actively worse
   than plain max, which would make LH harmful exactly where data is scarce.
3. **A packing defect** — E2M1 encoding, low-nibble-first order, or the `input_scale` convention
   (we set `input_amax/6`; if vLLM expects `amax/448` the activations are off by 74.7×, though the
   output is far too coherent for that).
4. **The gate/up split** — we assume `gate_up_proj` is `[gate; up]` along dim 1. Reversed or
   interleaved would corrupt SwiGLU systematically.

**Discriminating test:** rebuild one layer with plain max instead of Local-Hessian and re-probe. If
plain max is clean, the calibration is at fault; if it also corrupts, the packer is.
