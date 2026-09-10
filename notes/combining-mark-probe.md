# The combining-mark probe — a binary canary for quantization damage

## Why

Run A's readout is SWE resolution rate, whose SE is ≈2.9 points at 300 instances — it cannot see the
1–2 point effects at issue (`calibration-corpus.md`). Combining-mark corruption is a **discrete**
readout: a mark duplicates or it does not. No statistics, no slice size to argue about, and it targets
**ordering damage**, which is the failure mode a bad quant actually produces rather than a proxy for it.

## What it checks

The signature MiaAI reported in their #27 — duplicated **U+0E48 / U+0E49** in a copy task with MTP on —
generalised to five scripts that stress the same machinery: Thai, Devanagari, Arabic (with full
harakat), Hebrew (niqqud), and emoji ZWJ sequences. CJK is deliberately absent: no combining marks,
no signal.

`/opt/llm/runners/lh/markprobe.py`. Per response it reports exact-copy match and, separately, a
codepoint census **restricted to Unicode categories Mn/Mc/Me**. That restriction is the point —
self-tested 2026-09-10:

| input | flagged |
| --- | --- |
| clean copy | none |
| one injected duplicate U+0E49 | `{'U+0E49': 1}` |
| a dropped word (wording drift) | none |

so the probe separates "the model said something slightly different" from "the model duplicated a
combining mark". Only the second is the #27 signature. It also asserts `completion_tokens > 0` per
response, so an empty cell voids rather than scoring as a clean copy.

## What it does and does not tell us

**Does:** whether our Local-Hessian rebuild introduces ordering damage that the shipped checkpoint does
not have. The pairing is unusually clean — prod **is** `RadixArk/Qwen3.8-Flash-Next-NVFP4`
(`hf_quant_config.json`: ModelOpt 0.46.0 release, `*.linear_attn.*` and `*.self_attn.*` excluded), so
stock-vs-rebuild differ in exactly one thing: our expert weights.

**Does not:** say anything about MiaAI's #37. That hypothesis is about the GDN/linear-attention path
being quantised by a five-month-old ModelOpt. Our rebuild keeps RadixArk's exclusion set and touches
only `experts.<e>.{gate,up,down}_proj`, so on that axis our quant and RadixArk are the *same arm*.
A clean result here is also not a quality measure — it is a canary, and canaries only speak when
something is wrong.

## Run protocol

Both arms MTP on (prod n=3), matched sampling, temp 0, 3 reps per case, against the served endpoint.
Baseline arm should be folded into the next server start rather than paying a dedicated 9-minute load.
