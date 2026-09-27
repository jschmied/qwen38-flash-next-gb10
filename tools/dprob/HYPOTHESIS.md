FNDRAFTPROB (queued "probabilistic draft sampling for our head", user 2026-09-26), written 2026-09-27 ~03:50 before the run.
Change: draft_sample_method='probabilistic' (no local argmax); Qwen4ExpMTP.compute_logits returns the 32k NVFP4-slice
logits scattered into a -inf full-vocab buffer, so drafts are sampled from q = softmax(slice logits / T) and the ratio
test uses the same q (exact; q = 0 outside the slice). Arms: K=3 and K=5 (block 1728), greedy (base, local argmax) vs
prob, 2 starts, codeprobe (code greedy / code sampled 1.0-0.95-20 fixed seeds / prose greedy / code c4).
Sizing (5l): greedy->sampled acceptance gap 0.21 @K3, 0.46 @K5 (3.36->3.15, 4.35->3.89).
H1 greedy cells: temperature 0 -> drafts are the slice argmax either way -> hashes IDENTICAL to base, ms/tok within
   +0...+1.5 % (full-vocab gumbel/scatter/draft-logit cache per draft step). Hash mismatch = defect (void, triage).
H1 sampled code cell: accept_len K3 3.15 -> 3.20-3.30, K5 3.89 -> 4.00-4.20; ms/tok K3 -1...-3 %, K5 -2...-5 %.
H0: no acceptance gain (the slice renormalisation / missing top-k,top-p on q eat it) -> not worth carrying.
