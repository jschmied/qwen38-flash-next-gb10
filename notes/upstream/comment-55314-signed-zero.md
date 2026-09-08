DRAFT — needs the user's go. vllm-project/vllm PR #55314, first comment from us (2026-09-08).
Do not post until `zsign` lands: the set-level claim below is source reading plus arithmetic, and
the run that turns it into a measurement is queued. Placeholder marked ⟨ZSIGN⟩.

---

Two things from measuring this PR on a GB10 (sm_121, TP1), one supporting and one a defect.

**Your exactness fix works, and it is nearly free.** Built your branch standalone and ran it over an
81-shape grid: the selected *value multiset* matches the exact reference on every tie-heavy shape,
including the `num_ties` sweep that motivated #51782, at 1.00–1.15× stock. That is a real
improvement over stock and it is cheap.

**It does not make the kernel reproducible, and I think that is worth stating explicitly in the PR**,
because the two properties get conflated. Across 81 shapes × 6 calls, your branch is 0/81
self-consistent — the same input still returns a different *order* call to call, and two emission
sites still pick *which* tied element survives by arrival: the `atomicAdd(&shared_final_k, -1)` in
the last refine pass, and the `bp < MAX_BUFFERED_ITEMS` / `bp < DBUF` stash clip, which is reachable
at `level == 4`. On our model the ordering is what bites: the sparse attention sums the selected
keys in output order, so greedy decoding forks between identical requests even when the set is
exact.

**The defect: signed zeros are split across coarse bins, which changes the selected set.**

    convert_to_uint32_v2(+0.0f) = 0x80000000     convert_to_uint32_v2(-0.0f) = 0x7FFFFFFF
    convert_to_uint8(+0.0f)     = bin 128        convert_to_uint8(-0.0f)     = bin 127

`+0.0` and `-0.0` are numerically equal, so under "value descending, ties by index ascending" they
must tie. Both of your key functions rank `+0.0` strictly above `-0.0`, so on a row where the
top-k boundary falls inside a run of signed zeros the kernel returns the `+0.0` positions ahead of
lower-indexed `-0.0` positions — a different *set*, not just a different order. ⟨ZSIGN⟩

One line in each fixes it, before the order-preserving flip:

    if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;          // convert_to_uint32_v2
    if ((b & 0x7FFF) == 0) b = 0;                        // convert_to_uint8, on the fp16 bits

This also matters beyond the tie rule: the fp16-derived bucket is only useful because it is monotone
in the fp32 order, and the ±0 split breaks that monotonicity exactly at the pivot, which is the one
place it has to hold.

Happy to be wrong about any of this — the standalone harness and the shape grid are in
`jschmied/qwen38-flash-next-gb10` if you want to reproduce it. No ask attached; #55122 takes a
different approach to the same kernel and I am not proposing you adopt it here.

<!-- AI disclosure: this analysis was produced with AI assistance; I reviewed every line and ran
     every number quoted. -->
