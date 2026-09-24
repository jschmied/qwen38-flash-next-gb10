# FULL_DECODE_ONLY decode graphs on Flash-Next — hypothesis, written 2026-09-24 before any run

**Change.** Prod runs cudagraph_mode PIECEWISE, which captures nothing on this model (det-193/194), so every
decode step (target verify + 2 eager draft decodes) is launched kernel by kernel from the host. With the PLE
finalize fix (finding 236), FULL_DECODE_ONLY now captures. Sizes are [4,8,12,16,20,24,32,48,64], which covers
c=1..16 at 4 tokens per request under MTP n=3. In FULL mode the speculator also captures draft decodes.

**Prior:** the 08-27 profile measured the GPU 95.5 % busy at c=1 (no-spec, preview build); det-134 later measured
a 54 % duty cycle, later attributed to profiler overhead. Launch-bound share is uncertain, so the range is wide.

| cell | expected |
|---|---|
| FULL_DECODE_ONLY with prod's default compile mode | starts (if not: fall back to mode 0 with a PIECEWISE mode-0 control) |
| c=1 decode ms/tok, FULL vs PIECEWISE | −3 … −20 % |
| c=4 aggregate tok/s | +3 … +20 % |
| TTFT / prefill | unchanged (FULL_DECODE_ONLY leaves prefill eager) |
| c=1 output hashes | identical if the compile mode is unchanged; may differ under mode 0 (det-221) |
| acceptance | identical at c=1 when outputs are identical |
