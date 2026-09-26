K sweep on code vs prose (overnight 2026-09-26/27), written before the run. Stack: clone venv (branch 6b33c96 code),
mtpfp4 (FP8 GDN), RecoverSSM + align + prefix caching, NVFP4 draft head, capture sizes [1,2,4,5,6,8,10,12,16,20,24]
in every arm; K=5 needs --block-size 1728 (QSA ring capacity 12 must divide it; >= the 1696 auto size).
Why: bilikaz v4 reports 5.06 accepted/step at K=5 on a code prompt (73 tok/s); our probes are prose (2.5 at K=3).
H1 code c=1 greedy: accept_len K3 3.1-3.5, K4 3.6-4.2, K5 4.0-4.8. Step cost per extra verify row ~3.3 ms + draft
  step ~1.6 ms. Expected ms/tok code: K3 ~16, K4 ~15, K5 ~14.5 -> K5 wins code by 5-12 %.
H1 prose c=1: K3 21.3, K4 ~21.5 (+0...+3 %, finding 155: -3.4 % on agent prompts), K5 worse by 3-8 %.
H1 sampled code (1.0/0.95/20): acceptance 0.3-0.6 lower than greedy at every K.
Out of range: K5 code faster than 13 ms/tok or prose slower than 24 -> look for graph fallback / block-size effect.
