DRAFT — needs the user's go. ashhart/TensorFold #548 comment (2026-10-11).

A proposal for the shared kernels: let every prompt scan write its recurrent state at a few requested rows, the way Flash Next's already does. A family could then keep a prompt-cache state anywhere without ending a chunk there.

**What exists**

- Flash Next does this today. `pf_gdn_pre` writes the conv window at a mark row ([fn_prompt.metal L166-171](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/kernels/metal/prefill/fn_prompt.metal#L166-L171)). `pf_gdn_scan`/`scan4`/`scan8` write the state to `ST[m]` when `MK[m] == r + 1` ([L246-248](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/kernels/metal/prefill/fn_prompt.metal#L246-L248)). That allows up to `MARKS = 4` marks per call without ending the call ([marks.zig L10-11](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/src/families/flashnext/marks.zig#L10-L11)).
- Qwen 27B ends a chunk at each mark ([qwen27_slots.zig L47-49](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/src/native/qwen27_slots.zig#L47-L49)), so a mark costs an extra prompt call: one more pass over the weights. `warm` hides most of that by keeping the state in a background pass after the reply. A request that lands during that pass still pays for it.
- The other scans keep their state in registers and write it only at the end: `tf_delta_gdn_chain_wide` ([deltanet.metal L284](https://github.com/ashhart/TensorFold/blob/e7c523ebfa16b8094cf9e35e3773a7d25befcc5b/zig/kernels/metal/core/deltanet.metal#L284)), GLM's `glm_kda_scan`, and CUDA's `scan_rows.cu`. Nemotron and GLM are `planned`, so their marks floor to chunk starts.
- Nothing recurrent is shared yet apart from the cache itself. Each family has its own scan, its own conv-tail handling and its own Snapshots.

**What I'd propose**

One convention for prompt scans: a small mark list (`MK[4]`, the row after which to store, 0 for unused) and per-mark outputs for the state and the conv window. This is the Flash Next layout, kept bit-for-bit what a chunk ending at that row would leave. The cost is one extra store per mark row, and nothing changes when no marks are set.

Order, smallest first:
1. `core/deltanet.metal`'s prompt chain, the one shared DeltaNet library. Qwen 27B then keeps its marks inside a prompt call, as Flash Next does, and `qwen27_slots` stops cutting at them.
2. GLM's KDA scan and Nemotron's scans (Metal SSD and CUDA `scan_rows`), if you want those families off chunk-start-only marks.
3. A small shared helper for the snapshot copy (recurrent state, conv window, attention prefix), so each family only lists its layers.

Each step would carry a test: the state written at a mark is byte-equal to the state left by a chunk ending there, and a prompt resumed from it equals a fresh one.

I haven't measured the 27B side. I can't run the 27B here right now, so I'd need you or someone with a Mac to measure TTFT on a warm agent loop. Before writing step 1, does this convention suit you, or would you rather keep marks per family?
