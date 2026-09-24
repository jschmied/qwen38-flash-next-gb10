# Speed of light for Flash-Next decode on GB10 (step 1 of the waste audit, 2026-09-24)

**Question.** How far is c=1 decode from the floor set by the bytes it must move, and where is the rest?

**Method.** `tools/sol/headers.py` and `tools/sol/floor.py` (run from a directory where `headers.py` has written
`tensors.json`); output in `notes/data/sol-floor.txt`.
- Every tensor's dtype and size comes from the mtpfp4 safetensors headers; state and KV sizes come from
  config.json.
- One MTP n=3 verify cycle at c=1 reads:
  - the target once over 4 tokens: FP8 dense, the BF16 leftovers, the FP8 lm_head, and the routed experts
    actually touched;
  - GDN fp32 state, read plus written;
  - QSA selected K/V (≤ 2048 tokens) plus the compressed indexer keys;
  - the drafter 3×: MTP dense in BF16, the 45 MiB NVFP4 draft head, and experts, 10 per decode step.
- Floor = bytes / 220 GB/s, our measured bandwidth (det-231). At the 273 GB/s spec sheet every floor drops ~20 %.

**Sizes:**

| part | bytes per cycle |
|---|---|
| target FP8 dense | 2,551 MiB |
| target BF16 leftovers (+ norms, scales) | 1,912 MiB, of which hyper-connection mixers 1,209, shared experts 450, router 120 |
| lm_head (FP8) | 606 MiB |
| routed experts | E × 48 × 2.637 MiB (E = distinct experts per layer across the 4-token verify batch) |
| GDN state | 108 MiB read; 108–432 MiB written (1 or 4 speculative positions) |
| QSA | ~50 MiB at any context (selection-capped) |
| drafter, per cycle | 739–828 MiB (MTP dense BF16 173 MiB × 3, draft head 45 × 3, experts) |

**Floor vs measured.** Measured: 23.36 ms/tok × 2.535 = **59.2 ms per verify cycle** (finding 234, NVFP4 draft head,
c=1).

| E (distinct experts, 4 tokens) | floor ms/cycle | floor ms/tok | measured ÷ floor |
|---|---|---|---|
| 10 (fully shared routing) | 35–37 | 13.8–14.5 | 1.6–1.7× |
| 20 | 41–43 | 16.2–16.9 | 1.4× |
| 30 | 47–49 | 18.6–19.3 | 1.2× |
| 39.3 (independent routing) | 53–55 | 20.9–21.6 | 1.1× |

Context length barely moves the floor (1k → 32k adds ~0.3 ms), because QSA reads only its selection.

**Reading:**
- **The decisive unknown is E.** Near-independent routing puts us at ~1.1× the floor, so only fewer bytes help:
  - the BF16 leftovers at 8 bits save ~0.95 GiB/cycle ≈ −8 % (the INT8/MXFP8 lever, now sized);
  - the MTP dense layers at FP8 save ≈ −2 %;
  - writing GDN state for only the accepted position saves ≤ −2.5 %.
- **Correlated routing (E ≈ 10–20) would leave 30–40 % unexplained:** launch gaps, inefficient kernels, redundant
  work. The step-2 kernel audit targets exactly that.
- The routing capture used by det-198 (`/opt/llm/calib/capture`) no longer exists. **Step 2 must log live MoE
  top-k ids** alongside the kernel profile to pin E.

**Step 2 (planned, prod-down, after the FULL-graph A/B):** a torch profile of steady-state c=1 and c=4 decode, plus
logged routing:
- kernels mapped to modules (`kernelroster.py`, `prof_attrib.py`);
- each group's floor from its bytes, sorted into necessary-efficient, necessary-inefficient and unnecessary
  (copies, casts, repeated metadata, rejected-draft verify work, drafter re-prefill, graph padding).
