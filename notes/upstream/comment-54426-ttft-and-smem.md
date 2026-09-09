DRAFT — user go "yes, keep short, post". vllm-project/vllm issue #54426, comment (2026-09-09).

@rmagur1203 Three things on your TTFT result, one of which is a gap in our own corroboration.

**The shared-memory number behind your diagnosis.** GB10 reports **102400 bytes of shared memory per
SM** (verified on this box today, `sm_121`, 48 SMs). That is why an fp32 intermediate in `_cast_kv_tile`
forces the `block_n // 2` halving on this family — it is a hardware constant, not a tuning choice.
Unrelated but same number: FlashInfer 0.6.18 refuses its top-k path on GB10 because it wants ≥ 131072
(#55872, today). Two different kernels, one limit. If the per-tensor scale can be folded without the
fp32 tile, this family is where it pays.

**Careful with the acceptance delta.** We measured a **1-ulp** numerics change — a fused kernel within
one bf16 ulp of the original — move MTP acceptance **+6.6 / +10.4 / −3.8 pp across three prompts**, i.e.
expectation zero and a ±10 pp spread from rounding alone. Your −8.0 pts on one checkpoint and −2.7 on
the other sits inside that. Worth a per-prompt spread before it is booked as a cost of fp8 KV; we lost a
day treating an acceptance move as a speed effect.

**And a correction to our own earlier comment here.** We reported capacity (×1.72) and decode (+2.6 %,
n=6) and said "no decode regression" — we **never measured TTFT** on the fp8 arm, so our corroboration
should not be read as clearing prefill. On our own workload that is the expensive half: agent turns here
are TTFT-bound, 53–69 % of a turn, so a real +4.3 % TTFT would outweigh the pool gain for us. Your 9/9
cells in both checkpoints is more convincing than anything we posted on that axis.

*AI assistance was used in preparing this comment; the measurements are ours and were reviewed before posting.*
