DRAFT — user go given ("ok"). GitHub vllm-project/vllm PR #55122 (2026-09-08).

Two independent confirmations arrived overnight, from hardware and configurations that are not mine.
Recording them here because one of them names a bug in this PR's diff better than I did, and the other
widens the blast radius of the defect beyond what this thread has shown so far.

**1. A downstream project shipped this kernel and reproduced the launcher bug as a hard error.**
[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX) builds these sources at a
pinned commit and serves Qwen3.8-Flash-Next on a GX10. They bumped to the current revision and
[verified it on their own box before merging](https://github.com/blazux/qwen3.8-Flash-DGX/pull/10#issuecomment-5578884201):

> your new suite passes 210/210 with the new kernel while the previous pin fails case 79 with
> `chunk_size 256 smaller than TopK 512` — a hard error, so the launcher bug was a real crash waiting
> to happen at those widths

I had described that defect in the description as a sizing mistake — the chunk sized from
`sharedMemPerBlockOptin` rather than the opt-in minus the kernel's static `__shared__`, so 8 of 40
wide-row shapes could not launch. Their statement is better: it is a **named hard error, on a second
GB10, in an image that was already shipping**. Their independent numbers on the same commit:
micro-bench 1.0–2.4× vs stock (was 1.8–3.8× at the older pin), decode-sized rows ~1.0–1.1×, and at
the model level 4/4 prompts deterministic with no errors.

**2. A reproduction on a different model family and a multi-GPU arrangement.** @mmastrac reports on
[#54521](https://github.com/vllm-project/vllm/issues/54521#issuecomment-5578802318) the same class of
failure on **GLM-5.3-Flash across 4× DGX Sparks at TP=4** — responses stopping where tool calls should
be, plus what may be corruption during successful generation. Everything on this thread until now was
Qwen3.8-Flash-Next at TP=1 on one box.

**I want to be careful about what that second one does and does not show.** I have no TP=4 evidence
and do not run GLM-5.3-Flash, so I cannot claim their symptom is this kernel. I have asked them to run
the discriminator that would settle it — hash eight identical completions at TP=1, where more than one
class implicates the indexer's arrival-ordered slots and a clean TP=1 points elsewhere. That result is
worth having either way, and I will report it here whichever way it falls.

They also found something orthogonal that is worth knowing for anyone debugging tool-call loss on
these models: `validate_tool_names=True` in `glm47_moe.py` makes a tool call with an unrecognised name
emit zero deltas and finish `stop` with no content and no `tool_calls`, so corruption is
indistinguishable from the model declining to call a tool. That would mask this defect in any harness.

Nothing here changes the diff. The PR still has no reviewer and `pre-commit` is still gated behind a
label, so I am adding evidence rather than asking for anything.
