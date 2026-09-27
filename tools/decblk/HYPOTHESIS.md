DECBLK (vllm#53912 contamination check on our stack), 2026-09-27, before the run. User go: "ok".
Question: can a prefix-cache hit on a block whose recurrent state was written during MTP decode (with rejected drafts,
disable_eagle_block_drop=true) return state that differs from a fresh computation beyond numeric drift?
Arms (prod venv incl. the boundary fix, prod config, 4 GiB KV, block 1728 in both): spec (MTP K=5 + RecoverSSM + nodrop)
vs nospec (no speculation, native align checkpoints: the counterfactual that has chunking drift but no drafts).
Probe decblk.py: 16 seeds; ~1.4k-token random-number prompt, 520 sampled tokens at T=1.0 (low acceptance), follow-up =
prompt + 480 generated ids (crosses 1728); B = cache read (must hit >= 1728 tokens), R = cache skipped (prompt_logprobs=1,
must hit 0); greedy 24 tokens, compared.
H0 (RecoverSSM commits accepted tokens only -> no contamination): divergent(spec) <= divergent(nospec) + 2.
H1 (contamination): divergent(spec) clearly above nospec (>= +4 of 16), first divergence at token 0-2.
Void: B_hit_all false or R_miss_all false (then the probe did not test what it claims).
