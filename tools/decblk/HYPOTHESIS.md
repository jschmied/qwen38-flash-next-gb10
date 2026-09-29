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

AMENDED 2026-09-27 ~21:00 (decblk v1 VOID by its own rule: B had 0 prefix hits in all 16 seeds; the engine reported a 0.0 %
hit rate for the whole run on /v1/completions with token-id prompts, cause not found). v2 = decblk2.py on the chat endpoint
with continue_final_message (both requests render the assistant turn through the same template path) and a diag block
that records whether chat and completions hit the cache at all. Same hypotheses and void rule (B must hit past the first
request's prompt, R must miss).

AMENDED 2026-09-28 (decblk2 spec arm VOID by the same rule: 0 hits past the prompt in 16/16 seeds, while the diag's
identical repeat hits 3456 tokens; the nospec arm was still running when this was written - the mechanism below
predicts 0 there too). Cause found in the code, not guessed: align-mode Mamba keeps recurrent-state
snapshots sparsely - `prefix_cache_retention_interval` defaults to 0 = "only semantic checkpoints" (the latest replay
boundary and shared-prefix junctions; single_type_kv_cache_manager.py reachable-boundary mask). A block boundary crossed
during decode is not retained, so there is nothing for B to hit, with or without speculation. Consequence for prod
(interval unset): a decode-written recurrent state is never served from the prefix cache, so #53912's path is not
reachable in our config. v3 = decblk3: same probe, both arms with `--prefix-cache-retention-interval 1728` (every block
retained), to test the path for configs that do set it. Same H0/H1 and void rule.

AMENDED 2026-09-28 ~11:05 (decblk3 VOID at startup: "prefix_cache_retention_interval (1728) must be ... a multiple of
scheduler_block_size (3456)"). The scheduler block on this model is 3456 tokens, and cache hits come in that unit (the
diag's hit was exactly 3456). So v2's follow-up (1871 tokens) could never hit, whatever the retention rule: the v2 void
was the probe's length, not (only) retention. decblk3b: decblk3.py = 380 numbers (~3.1k-token prompt), 520 sampled tokens
cross 3456 during decode, follow-up ~3.57k; retention interval 3456. B must hit >= 3456 > prompt. Same H0/H1/void rule.

AMENDED 2026-09-28 ~11:30 (decblk3b VOID: prompt 3071, follow-up 3551; B hit 0 (spec) / 1728 (nospec), never 3456).
The 3456 boundary lies inside the generated tokens, and v2/v3 rebuild the follow-up by re-tokenizing the generated
text, so a mismatch there is not excluded. decblk4 = v1's exact token-id method (/v1/completions, return_token_ids,
follow-up = prompt ids + first 480 generated ids) with a 380-number prompt, interval 3456. If B still stops below 3456
with exact ids, decode-crossed boundaries are not retained even with the interval set: record that as the answer.

RESULT decblk4 (2026-09-28 11:55): valid. Exact token ids -> B hit 3456 in 16/16 seeds (past the 3047-token prompt,
i.e. a decode-written boundary state was served), R missed in all. Divergent: spec 4/16 (first at 19, 6, 8, 22), nospec
2/16 (7, 7). H0 holds (4 <= 2 + 2); H1 (>= +4 and first divergence at 0-2) not met: late, drift-sized divergence.
DECBLK5 (prod's default retention interval 0, spec arm only, same probe, 1 start). Code-reading prediction: the
decode-crossed 3456 boundary is not retained at interval 0, so B stops at a prompt-internal boundary (1728 or 0) in
16/16. Out of range: B hits 3456 -> the code reading was wrong and prod does serve decode-written states (then the
decblk4 H0 result is the relevant one for prod).

## decblk6 (2026-09-29, before the run): #53912 at prod's `--prefix-match-unit 64`

Trigger: cch-zuzuche on #53912 (2026-09-29) shows the fine-grained Mamba lookup (`prefix-match-unit` below the Mamba
page) returns the partial-tail entry at the prompt's last hash boundary and ignores `drop_eagle_block`; their corruption
is `!`-walls at 0 % acceptance on long reused agent prefixes. decblk5 (the "unreachable" result) ran WITHOUT
`--prefix-match-unit 64`, which prod has used since 09-28, so it did not cover this path. Same probe (decblk4.py, 16
seeds, token-id follow-up = prompt + 480 sampled ids), decblk5's spec arm + `--prefix-match-unit 64
--mamba-ssm-cache-dtype bfloat16`, plus a nospec control with the same flags; 1 start each.
H0 (clean): B hits ~3,008 tokens (the prompt's last 64-token boundary, a prefill-written state) in 16/16 of both arms;
divergent spec ≤ nospec + 2, any divergence late (token ≥ 5). H1 (poisoned): spec ≥ nospec + 4 divergent, or any
divergence at tokens 0–2, or a degenerate B. Void: B hits 0 in both arms (the path was not reached), or R hits > 0.
