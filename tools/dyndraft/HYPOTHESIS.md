# Dynamic draft length (FNDYN) — hypothesis, written 2026-09-24 before any run

**Change.** After each MTP draft token, take the drafter's max softmax probability over the 32k slice logits and keep
a running product. Stop drafting before the next step once every request's product is below the threshold. The
remaining draft slots repeat the last draft; the target verifies them normally, so output is exact. It costs one
host sync per draft step. Each draft step is an eager MTP-layer forward, so a skipped step saves real time.
Field: vcruz305 on exllamav3/GB10, up to 5 drafts with threshold 0.6, gave +27 % on prose and nothing on code.

**Design.** Threshold read from a file on every draft cycle. One server start sweeps 0 (off, control), 0.3, 0.5
and 0.7, then again in reverse, so the control sits within the same start. Arms are n=3 and n=4, two starts each.
The within-start `thr 0` rows carry the confidence computation but never stop.

**Expected.** An outcome outside these ranges means I debug the instrument first.

| cell | expected |
|---|---|
| c=1 output hashes, every threshold vs 0 | identical (the target verifies every draft) |
| acceptance length vs thr 0 | lower, because stopped cycles offer fewer drafts: −5 … −25 % at thr 0.5 |
| c=1 decode ms/tok, n=3, best threshold vs thr 0 | −2 … −10 % |
| c=1 decode, n=4 best vs n=3 best | −3 … +5 % (static n=4 was −3.4 %; gating removes most of its cost) |
| c=4 decode | smaller effect than c=1; a batch stops only when all 4 are unconfident |
| the one host sync per draft step | should not dominate: thr 0.7 with no stops would show pure overhead |
