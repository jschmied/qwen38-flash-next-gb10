**Correction to lever 3 in the issue body: I ran the falsification test I proposed there, and it kills
the idea. NVFP4 is not losslessly compressible.**

I wrote that per-group scaling should leave the extreme codes rare, so empirical entropy might be
3.0–3.5 bits and buy 12–25 % less traffic at bit-exact quality. Measured on 73.7 M nibbles sampled
across layers 0/12/24/36/47, three experts each, all of gate/up/down:

| | |
| --- | --- |
| order-0 nibble entropy | **3.969 bits of 4.000** — 0.8 % below FP4 |
| order-1 conditional (the two nibbles sharing a byte) | 3.968 — correlation negligible |
| most common code (−2.0) | 7.56 %, against 6.25 % for uniform |
| `weight_scale` FP8 E4M3 stream | 5.138 of 8 bits — 35.8 % below, but it is 1/16 of the bytes |
| **combined lossless floor** | **2.8 % smaller than packed NVFP4** |

2.8 % does not justify a decoder, and it certainly does not justify one that has to clear ~190 GB/s.

**The reason is structural, and I should have seen it before proposing it.** Per-group amax scaling
exists precisely so each group of 16 uses the full code range. Near-uniform code occupancy is not an
accident — it is the evidence the quantizer is doing its job. Any format with a well-fitted group
scale will behave this way, so this result should generalise to your CB3 packing too: expect its
3.07 bpw to be near *its* entropy floor as well, i.e. the win there is the lossy step, not any
residual redundancy on top.

Consequences for the list in the issue body:

- **Lever 3 is withdrawn.** The bytes are irreducible without a quality trade.
- Levers 1 (fetch/compute overlap and batching) and 4 (more NVMe) are unaffected.
- Lever 2 (per-layer arena allocation) — see the separate note below; I also tested that, and it did
  not survive either.

**Separately, on paging policy.** Since the issue body touched the arena, and a segmented cache with
TinyLFU-style admission is an obvious next step: I replayed real `(layer, expert)` sequences from our
own model against a Belady/MIN oracle before writing any policy, 7.2 M scored accesses, 25.6 %
resident, frequency policies ranked on a held-out prefix rather than in-sample:

| policy | hit |
| --- | --- |
| static trace-ranked | 55.4 % |
| **global LRU** | **90.1 %** |
| decayed LFU (sampled eviction) | 88.5 % |
| protected static set + admission on 2nd use | 84.8 % |
| Belady (full oracle) | 95.6 % |

**Nothing beat plain LRU**, and a frequency-ranked protected segment was 5.3 pp *worse* than LRU — it
locks slots to experts that are not preferentially reused. Our routing entropy is 8.96 bits of a
possible 9.00 (perplexity 496 of 512), i.e. near-uniform, so there is no global hot set to protect and
any real hit rate must come from temporal locality, which LRU already captures.

The oracle gap is 5.5 pp of hit, which is 55 % less NVMe traffic — worth chasing, except that it is
not reachable cheaply: Belady restricted to a drafter-sized lookahead recovers **2.3 % of the gap at
1 token, 5.7 % at 3, 10.9 % at 6**. Speculation does not act as a partial oracle; Belady wins on
knowing what will *not* be needed for thousands of accesses.

So your existing design — trace-ranked warm start, then one global LRU — looks right, and the honest
reading is that the warm start is worth much less than the LRU that follows it.

**Caveats.** All of this is Qwen3.8-Flash-Next: 512 experts top-10 over 48 layers, calibration text,
no prefill or session boundaries in the sequence. Yours is 384 top-6 over 40 layers on agent traffic.
The *gaps between policies* are what I would carry across; the absolute hit rates are not portable,
and they move with sequence length in our own runs (90.1 % at 20 k tokens, 93.5 % at 4 k). The entropy
measurement is on NVFP4 specifically and is the more transferable of the two.

Replay harness and the raw numbers:
[notes/data/](https://github.com/jschmied/qwen38-flash-next-gb10/tree/main/notes/data) —
`beladyreplay.py.txt`, `belady-replay.txt`.

Drafted with AI assistance; every number re-derived and checked before posting.
