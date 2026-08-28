# Implementing fused multi-step draft decode for QSA — it works, and it buys nothing

vLLM logs this on every Flash-Next start with MTP:

```
[speculator.py:117] Fused multi-step draft decode is not supported by attention backend(s)
QWEN38_FLASH_NEXT_EXP_QSA_STATE; falling back to rebuilding attention metadata between draft steps.
```

Nobody upstream has implemented it — no vLLM issue, no PR, no mention in #53896. We did, to find
out what it was worth. Patch: `patches/qsa-fused-draft-decode.patch`.

## The contract is two things

A backend sets `supports_draft_decode_metadata_update = True` and implements
`update_draft_decode_metadata(metadata)`. Four backends do: `triton_attn` (whose implementation is
literally `pass`), `flash_attn`, `mla/triton_mla`, and `mla/sparse_swa` — the last being sparse
*and* stateful, i.e. the same shape of problem as QSA, and the template we followed.

## Why it can be done cheaply

QSA metadata is genuinely position-dependent, so unlike `triton_attn` it cannot be a no-op:
`logical_positions[t] = seq_lens[req] - query_len[req] + within_query[t]`, and `slot_mapping`
derives from that. But three facts make a re-run of the *existing* kernel sufficient:

1. The speculator advances `positions` and `seq_lens` **in place** in `input_buffers`
   (`_update_draft_inputs_kernel`, `ADVANCE_DRAFT_POSITIONS=True`).
2. `build_attn_metadata` stores `seq_lens[:num_reqs]` and `slot_mappings[i]` — **views**, not
   copies — so a cached `CommonAttentionMetadata` sees the advanced values.
3. `build_qsa_metadata_triton` writes **in place** into the persistent `token_to_req` /
   `logical_positions` / `slot_mapping` buffers that the live metadata already points at.

So `update_draft_decode_metadata` caches the common metadata at `build()` time and re-runs the
same kernel. Reusing the identical kernel rather than hand-writing an update means the values
cannot diverge from the rebuild path — only the scaffolding around them is skipped.

## It is correct

Acceptance is the correctness gate: wrong positions make drafts stop being accepted rather than
crash, and output stays coherent regardless because speculation is verified against the target.

| c=1, i4000/o512 | decode t/s | acceptance | pos0 | pos1 |
| --- | --- | --- | --- | --- |
| stock | 36.2 | 56.6% | 67.3% | 46.0% |
| patched | 34.1 | 54.9% | 65.9% | 44.0% |

Unchanged within the variation from different random corpus slices.

## It buys nothing

**At k=2 the patch is a no-op by construction.** The guard is

```python
if step < self.num_speculative_steps - 1 and ...:
    attn_group.update_draft_decode_metadata(attn_metadata)
```

At k=2 the loop runs once with `step=1`, and `1 < 1` is false — the update never fires, and both
paths do exactly one build plus one draft. Any earlier claim that k=2 "pays a rebuild the fix
would remove" was wrong.

At k=3, where it does fire (0 fallback messages, update active):

| c=1, i4000/o512 | decode t/s | c=16 aggregate |
| --- | --- | --- |
| stock k=3 | 36.8 | 100.5 |
| patched k=3 | 36.1 | 100.1 |

**Below run-to-run noise** (stock k=2 alone spans 36.2–38.0 on identical settings). The metadata
rebuild was never the bottleneck, so the premise for the whole exercise was wrong.

## What the experiment did find

k=3 per-position acceptance, measured for the first time: **68.0 / 46.0 / 33.1%**.

| | tokens per iteration |
| --- | --- |
| k=2 | 2.133 |
| k=3 | 2.471 |

k=3 yields **15.8% more tokens per iteration for identical throughput** — so the third draft step
costs almost exactly what it returns. Each draft forward is ~15% of an iteration while the drafter
reads only ~0.7 GiB against the target's 6.13 GiB: 11% of the bytes for ~15% of the time.

**The draft forward is what caps speculation here, not the metadata.** The drafter's MoE is
`mtp.layers.0.mlp.experts.*` — 4.86 GiB of unquantized BF16, 2 tensors, no scales, sitting beside
294,912 properly scaled body tensors. That is the lever, and it is the same one that would unlock
the MoE backend set (see `moe-backend-axis.md`).

## Should it go upstream?

The patch is correct and closes a real gap, but it delivers no measurable gain on this hardware
and configuration. Reporting the *finding* — that the rebuild is not the cost — is worth more than
the patch. Kept here rather than pushed.
