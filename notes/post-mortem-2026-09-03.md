# Post-mortem — the MTP "acceptance flip" investigation (2026-09-01 … 09-03)

## What was actually found

| # | defect | evidence class | status |
| --- | --- | --- | --- |
| A | align-mode block size = QSA ring capacity (16) instead of the mamba block (1616): every prefix-cache resume seeded the running state in the wrong units | 7 starts + replays, per-turn; code confirmed | real; PRs #54076/#53798 applied on this build (finding 46) |
| B | `persistent_topk` on sm_121 returns a varying order and sometimes a varying set for identical inputs → long-prefill hidden states differ per request | bit-level: 3 identical 7.5k-token requests, every layer/submodule hashed (52–54) | real; exact `torch.topk` at the call site makes the forward and the MTP loop bit-identical (54, 55, 58) |
| MoE | FlashInfer CUTLASS NVFP4 fused finalize nondeterministic; no vLLM switch | bit-level, 3 shapes (37) | real; vllm#54945 / PR #54948 |
| "C" | per-turn / per-depth low acceptance that survived every fix | — | **an artifact of the benchmark** (59) |

## The artifact, and why it survived 58 findings

`agentloop*.py` sent `ignore_eos: true, max_tokens: 130`. The model answers the loop's prompts in
30–40 tokens; the remaining 90–100 tokens of every turn are what the model does *after* its
end-of-turn token: either a chat-template restart plus a verbatim repeat of the answer (drafts
accepted ~100 %) or a wall of `<|im_start|>` (drafts accepted 0 %). "Healthy" and "broken" turns
were these two fillers. Which one the target picked was a near-tie after EOS — random per request
while A/B made the numerics nondeterministic, fixed per depth once they were deterministic.

Why nobody saw it: the loop reported tokens, ms/tok and acceptance counters, never text. Each of
those looked exactly like a real per-request defect (two clean states, lifetime-stable,
prompt-independent under nondeterminism, not a config knob, not a race, not the ring, not the
target's *text* — 45 compared texts against a no-spec reference that had the same filler). The
first instrument that showed text alongside acceptance (59) exposed it in one look.

The `ignore_eos` choice was deliberate: it pinned the work per turn so ms/tok was comparable
between arms ("s/turn alone is meaningless if turns differ in length"). It pinned the wrong thing.

## What that contaminates

Every MTP acceptance, accept-length and ms/tok number produced by `agentloop*.py`, i.e. findings 6,
8, 9, 40, 41 (the *rates*), 43, 46 (the *rates*; the mechanism stands), 55, 57, 58; the per-depth
"anomalies" in `mtp-depth-anomaly.md`, `depth-curve.md`, `mtp-vs-prefix-cache.md`,
`which-drafter-for-agent-work.md`; and the audit's MTP rows. The upstream comments on #53142 /
#54076 / #53798 quote healthy-turn percentages from this loop — the *direction* (the align fix
matters for MTP) survives because the fix changed which filler was picked, but the numbers must be
re-stated from the EOS-correct grid (`MTPGRID3`) before anyone builds on them.

What is not contaminated: every bit-level result (hashes with `max_tokens` 1–4, layer bisections,
the exact-top-k proof), the MoE finding, the ring-capacity PR, the cache-key backport.

## Other things that went wrong

1. **Two env vars in one systemd `Environment=` entry** silently ran an arm without the fix under
   test (finding 56, withdrawn). Rule now in `failure-modes.md`: one variable per entry, and every
   arm must print and grep its own activation line.
2. **Single-start claims.** 40 published numbers rested on one start (see `evidence-audit.md`);
   the user's rule — no call from three runs — was applied only from finding 43 on.
3. **"Deterministic forward pass" was proved on a 55-token prompt** and generalised. Bug B needed
   7.5k tokens to show. The audit flagged it before the run that found it.
4. **Three bisections were run on top of an unfixed lower layer** (drafter hashes on the unpatched
   align path, then without exact top-k) and could not bisect anything; each cost ~40 min.
5. **The reference-text comparison (45) compared filler against filler** and concluded "the target
   is fine" — true, but for the wrong reason.

## Rules to keep

- Never benchmark speculative decoding with `ignore_eos` on a chat model; stop at EOS, report
  real tokens, and print the text next to the counters at least once per new instrument.
- A per-request state with two clean levels: look at the text before looking at kernels.
- Prove determinism at the longest context you serve, not the shortest that is convenient.
- One env var per `Environment=` entry; activation line per arm; three starts before a rate.
