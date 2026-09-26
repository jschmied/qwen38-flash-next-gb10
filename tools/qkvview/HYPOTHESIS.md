A1 FNQKVVIEW (speed-of-light 5f), written 2026-09-26 before the run.
Change: under RecoverSSM spec verify, q/k/v are strided views of the conv output (no CatArray + 3 direct_copy per
GDN layer) and verify writes into core_attn_out (no 49,152 B D2D memcpy). 180 launches/step removed, ~12 MiB/step.
Unit check: output and replay record bit-identical (batch 1/2/4).
H1: c1 per verify cycle (ms/tok x accept_len) -0.3 ... -0.8 ms (-0.6 ... -1.5 %), c4 tok/s +0.5 ... +1.5 %.
H0 range: |delta| < 0.3 % -> eager-gap removal does not transfer (like finding 237).
Must hold: c1/c4 output hashes identical to the base arm (same kernel, same values). Different hashes = defect, VOID.
Out of range (> 2 % gain or any slowdown > 0.5 %): look for another moved variable before believing it.
