@bojiang3 — linking this to the issue you looked at, since the two threads never got connected: this PR is the fix for [#54552](https://github.com/vllm-project/vllm/issues/54552), where you confirmed the arithmetic on 2026-09-02 and agreed that widening `capacity` to the next multiple of `compress_ratio` dividing `block_size` preserves the invariant. It has been open since that day.

Two things have been added since:

**Runtime evidence.** I applied the widening on a DGX Spark (GB10, sm_121, TP1) serving Qwen3.8-Flash-Next NVFP4 with `num_speculative_tokens=5`. It fired on all 12 QSA layers and cleared the assert:

```
QSACAP-PATCH: QSA ring capacity 12 -> 16 so it divides block size 1616
              (span 9, invariant capacity >= span holds)
```

**The widening is bounded.** A review follow-up caps it at 2x the minimal ring, because without a cap `num_speculative_tokens` 13..16 takes a 20-row ring to 212 rows on block size 848 and 404 on 1616, and every request holds a ring block for its lifetime. The intended 5..8 band (12 → 16) is unaffected.

Also worth flagging for whoever picks this up: on the V1 runner this assert is not the only thing in the way. Clearing it lands immediately on `RuntimeError: PLE inputs were not prepared` — reported independently in [#56088](https://github.com/vllm-project/vllm/issues/56088). So this PR is necessary but not sufficient for anything that forces V1; it is sufficient for the V2 paths, where the assert is simply unreachable-by-design today.

One process question: CI here shows `pre-run-check` failing at 5 s, which I believe is the label gate rather than anything in the diff. Is there a label needed to let the suite run?
