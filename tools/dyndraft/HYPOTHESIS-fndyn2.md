
# FNDYN2 (dynamic draft stop on the NVFP4 draft head), written 2026-09-26 ~13:45, before the run
Clone venv + fndyn2_patch.py; both arms prod-like (RecoverSSM immediate, NVFP4 head, 32k vocab, MTP n=3), KV 4 GiB,
comboprobe2, 2 starts. Arms: base vs dyn (FN_DYN_DRAFT=1, FN_DYN_DRAFT_THR=0.3). Finding 235 on the BF16 head:
-1.7 % c=1, +2.5 % c=4.
- H-speed: c=1 -0.5..-1.5 % ms/tok (smaller than 235: the NVFP4 head made each draft step cheaper, so a skipped step
  saves less); c=4 +0..+3 % (worse, as in 235); agent loop within +-2 %.
- H-correct: output hashes identical to base (the target verifies every token; only draft counts change).
- Outside: hashes differ -> the stop path changes verify batch shapes in a way that alters numerics (report, drop).
