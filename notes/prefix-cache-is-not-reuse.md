# The prefix-cache effect is not caused by cache reuse

`--no-enable-prefix-caching` made the prefill probe deterministic (`F_noprefix`: 1 distinct of 3,
`lp=-0.2308074981`, `sig=e22e0de36cac` three times). The obvious reading — "reusing cached state
returns something different from recomputing" — **cannot be right here.**

## The probe cannot produce a cache hit

- attention **block size is 1616 tokens** (log: *"Setting attention block size to 1616 tokens to
  ensure that attention page size is >= mamba page size"*)
- the probe prompt is ~50 words, on the order of **60 tokens**
- prefix caching registers only **completed** blocks
- sub-block hits would need `mamba_partial_cache_hit`, which requires
  `hash_block_size < block_size`, which requires `--prefix-match-unit`. That option defaults to
  `None` and we never set it, so partial hits are **off**

So the prompt fills 0 of 1616 tokens of one block, no hash entry is ever registered, and requests
2 and 3 cannot hit anything.

## What the flag actually removed

Not reuse — there was none. It removed:

1. **`mamba_cache_mode`: `align` → `none`.** The flag switches it as a side effect; `align` is the
   default whenever prefix caching is on. This is a second change hidden behind one knob.
2. **The align state machinery itself**, which runs regardless of hits: checkpoint writes,
   copy-on-write into private blocks (`_producer_partial_tail_reqs`, `last_state_block_idx`), and
   `postprocess_mamba_align_gpu` — a fused GPU postprocess that mixes state copies with
   accepted-token updates **without a CPU-GPU sync**.

**So the divergence comes from the machinery being active, not from anything it hands back.**
That removes a whole family of hypotheses — stale cached state, wrong restore, truncated
checkpoints — because nothing is ever restored in this probe.

## Consequence for the queued arm

`M_all` vs `M_align` keeps prefix caching **on** in both arms, so the machinery stays active in
both; only the checkpoint *frequency* differs. Under this analysis both should still diverge —
which is also what [vllm#54173](https://github.com/vllm-project/vllm/issues/54173) reports
independently (*"'align' and 'all' both fail identically"*). If both diverge, the cause is the
common path — CoW plus the unsynchronized fused postprocess — not the checkpoint policy.

The sharper experiment is therefore **prefix caching on with speculation off** versus
**on with speculation on**, since `postprocess_mamba_align_gpu` takes a different path when
`speculative_config is not None`. Note `P_nospec` already diverged with the cache on and spec off,
so the machinery alone appears sufficient.

## Status: MEASURED, not just derived

The servers log it directly:

| arm | prefix cache | logged hit rate | prefill result |
| --- | --- | --- | --- |
| `P_ctl` | **on** | **0.0%** | diverges 3/3 |
| `P_nospec` | **on** | **0.0%** | diverges 3/3 |
| `F_noprefix` | off | 0.0% | **deterministic** |
| `AC1_n5` (8-turn agent loop) | on | **55.3%** | — |

**The probe arms diverged at 0.0% hit rate.** Nothing was reused, so reuse cannot be the cause.
`AC1_n5` is the control that keeps this from being a broken-cache story: with accumulated context
the same build hits 55.3%, so 0.0% on a 60-token prompt is the expected consequence of block size
1616, not a defect.

Measured 2026-09-01. Related: `prefill-divergence.md`, `batch-invariance-unavailable.md`,
vllm#54173, vllm#47861 (closed; only its scheduler half merged via #51113).
