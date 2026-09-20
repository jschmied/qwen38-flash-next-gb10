DRAFT — needs the user's go. vllm-project/vllm PR #54076 (2026-09-20).

Following up on my 2026-09-17 withdrawal: the box has the checkpoint again, so I went to run the
cell you asked for. **It cannot be produced on this build, for the reason you already reported.**

**Box**: DGX Spark, GB10, sm_121, TP1. vLLM `0.28.1rc1.dev524+g5db652225` (base 2026-09-08) plus the
#53899 PLE-offload port, `Qwen4ExpForConditionalGeneration`, NVFP4 with an FP8 head, `--max-model-len
32768`, prefix caching on.

### The heterogeneous geometry never arises here

Two boots, one variable:

| arm | log line | resulting grids |
|---|---|---|
| no speculation | `interface.py:933` attention block → **1568**, `:957` pads the mamba page **0.13 %** | equal |
| MTP `num_speculative_tokens=3` | `interface.py:933` attention block → **1600**, `:957` pads **0.25 %** | equal |

The block size differs *between* arms but in each one `:957` pads the mamba page so the two are
"exactly equal", so `cache_config.block_size == MambaSpec.block_size` always and the split this PR
corrects is never reached. That is your 2026-09-09 result — "the KV-config interface normalization
already guarantees `cache_config.block_size == MambaSpec.block_size`" — reproduced on a third
configuration (GB10/sm_121, different model, different quantization).

**The `--block-size` escape does not work either.** `interface.py:931` is
`if cache_config.block_size < attn_block_size: cache_config.block_size = attn_block_size`, i.e. a
floor, so an explicit `--block-size 816` is silently raised to 1568 rather than creating the
divergence. On this build the regime the PR governs is unreachable from the CLI.

So I cannot give you a patched-vs-unpatched hit rate that means anything: both arms would be the
same engine. Rather than post a null and call it a result, here is the negative finding with its
mechanism. It does not argue against the PR — your defence-in-depth framing for explicit
`--block-size` and other page layouts is untouched by it, and the scheduler invariant is worth having
regardless. It does mean the number you asked me for is not obtainable on stock config for this
model.

### The diff no longer applies to current-ish main, and a rebase

For what it is worth on the rebase you have been doing repeatedly: against our `dev524` base, hunk 1
fails because the `MambaSpec` import it adds is already present (line 59, used for
`prefill_checkpoint_alignment`), and hunk 4 fails because it predates #53614's internal-checkpoint
exemption. Reconciled to your 2026-09-06 form — `0 if use_internal_checkpoint else
next_block_boundary` — it applies as 3 hunks / 48 lines. Happy to send that as a patch if it saves
you a round; I have not opened anything.

### Two measurement notes, since you asked for a hit rate specifically

- **The first repetition does not hit.** On this model, one prompt sent three times gives
  0 / 18,949 then 0 / 18,949 then **16,000 / 18,949**. A cold → one-re-ask harness therefore reads
  zero at a position where zero is correct. I built exactly that harness first and it cost me a
  wrong conclusion on an unrelated PR before I caught it.
- **`cached_tokens` stayed inert**, as flagged on 09-17: `None`/0 on every request, including ones
  with ~9,000 cache queries. `vllm:prefix_cache_hits_total` deltas are what I read.

**Not measured**: anything on a build newer than 2026-09-08, and no forced-`--block-size` arm, since
the floor above makes it a no-op rather than a heterogeneous layout. If you can name a configuration
in which the divergence *is* reachable on current main, I will run the original cell on it — the
hardware is free and a boot costs about 12 minutes.

_AI assistance (Claude Code) was used for this analysis; every number was checked against the run
that produced it._
