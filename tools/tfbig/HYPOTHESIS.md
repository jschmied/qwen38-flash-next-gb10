# End-of-work suites on TensorFold (2026-10-01, user: "skip queued swt ... run the other tests"), before the runs

Our checkpoint `qwen38-flash-next-mtpfp4`, `--context 262144 --parallel 8`, `longmix.py` (4 × ~120k-token prompts at
t=0, 8 × 2k every 15 s, 300 / 200 tokens), one start per arm, nothing else on the box.
- **Stock 0.6.0:** no out-of-memory stops, MemAvailable never below the 2 GiB reserve; the short requests wait behind
  the long prompts (~480k tokens at ~1,300 tok/s): first tokens **250–400 s**; the n-gram table is not locked on this
  checkpoint, so major faults grow during the long prefills (**≥ 10k** over the run).
- **`pr-ngram-gather` (T1 + T2 only):** on this checkpoint T1 does nothing (its table is FP8, not EXL3) and T2's
  fill acts on decode-sized gathers while the table is cold. If the run makes the table cold, the long streams' decode
  rate rises **0…+15 %** and major faults stay similar (the fill reads the same pages, in parallel); TTFT within ±5 %.
  If the table never goes cold (few faults), the arm is null — that is the result, not a failure.
- **`pr-fill-order` (#174):** short requests' first tokens **5–20 s**; the long prompts' first tokens later by ≤ 8 × 3 s.
- **Full `prefill_cold` 2k–65k with `--prefill-fp8`** on the same checkpoint (vs §5an's bf16 sweep): **1.12–1.20×**
  faster at every length.
