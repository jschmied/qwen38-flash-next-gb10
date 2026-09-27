F1 FNDRAFTPW (fusion plan item 1), written 2026-09-26 ~23:55 before the run.
Change: MTP draft decode steps 2..K run their compiled regions as PIECEWISE cudagraphs instead of eager
(stock: "PIECEWISE cudagraphs are not supported for draft decodes" -> NONE). Same kernels, same values.
Audit (5f): drafter steps 2-3 idle ~0.30 ms/cycle at K=3 from 171 eager ops; ~2x that at K=5.
H1: per cycle -0.15 ... -0.30 ms at K=3 (-0.3 ... -0.6 %), -0.3 ... -0.6 ms at K=5 (-0.5 ... -1.0 %); c=4 similar or
    slightly more; output hashes identical to the base arm in every cell (greedy cells; the sampled cell too, fixed seeds).
H0: capture fails / hashes differ -> PIECEWISE draft decode is unsafe here (void, triage).
