F4 (full CUDA graphs for the RecoverSSM verify path), 2026-09-28, before the run. User: "1. yes, do 4. next", then "all".
Builders gdn_recoverssm / ple_recoverssm report UNIFORM_BATCH (was NEVER, which forced PIECEWISE); padded graph rows
get null state slots and zero-length windows (kernel writes zeros and returns for state_idx <= NULL_BLOCK_ID).
Clone venv rssm, prod config (K=5, probabilistic drafts, bf16 SSM state, prefix-match-unit 64), KV 4 GiB, fastload on.
Arms: full (FN_CG_MODE=FULL_AND_PIECEWISE) vs pw (PIECEWISE), 2 starts each, alternating. Probe nvprobe.py.
H: greedy hashes identical between arms (same kernels, only launch changes); c=1 code/prose ms/tok -0 ... -4 %
(finding 237: FULL_DECODE_ONLY on the native path was null; c=1 duty cycle 54 % leaves launch gaps, but PIECEWISE
already graphs most of them); c=4 -0 ... -3 %; TTFT unchanged (+-2 %, prefill is PIECEWISE in both).
Out of range: any hash difference (padding/state bug), a slowdown > 2 %, or a gain > 6 % (then check what else moved).
Tests first: GDN/PLE/config RecoverSSM tests incl. the new padded-row test; a failure skips the A/B.

F4CTX (2026-09-28 ~11:20, after §5z: short-context c=1 -1.1..-2.1 %, but an 8k-prompt request +13..29 % slower with full
graphs). Same arms, 2 starts. Probe ctxprobe.py: decode ms/tok by streaming (prefill excluded) at ~1k / 8k / 16k / 28k
tokens of context, 2 requests each, fixed prompts so hashes compare across arms.
H (context-dependent cost under full graphs, e.g. attention/QSA launched for the capture shape, not the live context):
full <= piecewise at 1k (-0..-3 %), full slower at 8k (+5..+30 %), gap growing with context (28k worse than 8k).
H-alt (not decode): the §5z gap came from the request's prefill/scheduling (cold 8k also +7..13 %); then the streamed
decode ms/tok shows no context-growing gap (within +-2 % at every size) and TTFT carries the difference.
Hashes must match between arms at every size.

F4REP (2026-09-28 ~11:45, after f4ctx showed no decode-vs-context cost). The §5z gap was one request type: 8k prompt +
96 tokens (no ignore_eos), run after the TTFT probes; +0.35..0.5 s in both full starts. Probe replayprobe.py, same arms,
2 starts: 3 replay pairs (cold, cache-hit) on the fresh server (A), nvprobe's TTFT probes, 3 pairs again (B); streaming,
tokens, finish reason, hash, cached tokens.
H1 (output length): the arms emit different token counts/texts for this request -> times differ per token count only;
per-token decode and TTFT equal. H2 (order/state): the gap appears only in B (after the 30k probes), not in A -> a state
left by the long prefills under full graphs (graph pool / allocator / cache). H3 (real per-request cost under full
graphs, e.g. first decode steps after a prefill): gap in A and B with equal tokens and hashes, +0.3..0.5 s per request.
