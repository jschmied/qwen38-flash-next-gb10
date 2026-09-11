> **Revised 2026-09-11, shortly after posting.** The original version of this issue claimed
> expert-major repacking was worth "up to ~2.2×" and blamed the small-read end of the curve on the
> drive's DRAM-less design. **Both claims were wrong and are withdrawn** — my own table contradicts the
> first, and the second is an artefact of how I generated queue depth. The measurements are unchanged;
> the conclusions drawn from them are. What changed is marked below.

Measured on a GB10 box's NVMe with `O_DIRECT` on real safetensors shard bytes, so the page cache
cannot flatter the numbers. Drive is a **Phison PS5027-E27T** (PCIe 4.0 ×4, DRAM-less with HMB;
`ESL01TBTLCZ-27J2-TYN`, 1 TB) — likely the same part you have.

## The result I would actually rely on

**Sequential and scattered are indistinguishable at expert size.**

| pattern | GB/s |
| --- | --- |
| 18 MB contiguous, marching through one 10 GiB file | **6.37** |
| 18 MB scattered across 206 files | **6.42** |

At expert granularity, placement and seek locality cost **nothing measurable** on this SSD. Transfer
size and queue depth dominate.

And we are close to saturated: Phison rates the E27T at **7.4 GB/s** sequential read, so 6.4–6.5 is
**~87.8 %** of the controller's ceiling (82.5 % of the PCIe 4.0 ×4 encoding maximum of 7.877 GB/s).
The remaining 12–14 % could be the OEM NAND/firmware configuration, attainable queue depth, controller
or NAND limits, thermal state, or host overhead — **these measurements do not isolate which**, and 7.4
is a controller capability rather than a published rating for this exact OEM part (other E27T 1 TB
references give 7.35; some commercial products are rated 7.0).

## The size/depth curve

| read size | QD1 GB/s | QD4 GB/s | depth gain |
| --- | --- | --- | --- |
| 64 KiB | 0.46 | 0.99 | 2.2× |
| 256 KiB | 1.07 | 2.57 | 2.4× |
| 1 MiB | 2.40 | 5.70 | 2.4× |
| 4 MiB | 4.02 | 6.48 | 1.6× |
| 9 MiB — half an expert | 4.53 | 6.47 | 1.4× |
| 18 MiB — one expert | 5.06 | 6.42 | **1.27×** |

Queue depth alone saturates at **QD2** for expert-sized reads.

⚠️ **Important caveat on the small end.** Queue depth here came from a **thread pool, not io_uring**.
At 18 MiB each read is ~3 ms and thread overhead is noise, so the 1.27× stands. At 64 KiB each read is
~140 µs, where submission cost, scheduling, synchronisation and *effective* queue depth all matter —
so **the small-read rows understate what io_uring would deliver** and should not be read as a property
of the drive. (0.46 GB/s at 64 KiB is only ~7,000 IOPS, trivial against an E27T's ~1.0–1.2 M 4K
random-read IOPS, which is why I no longer attribute it to the DRAM-less FTL. An io_uring or fio sweep
is needed before blaming hardware.)

## WITHDRAWN: expert-major repacking is not worth ~2.2×

The original issue argued your ~2.5 GB/s was a layout symptom, since each expert is stored as a
scale-run plus a separate weight-run. **My own table does not support that**:

- combining two half-expert reads into one: **−0.8 % at QD4**, **+11.7 % at QD1**;
- and a real NVFP4 expert splits asymmetrically — roughly 17.7 MB of weights plus 1.1 MB of scales —
  so repacking merges a long read with a short one and saves **single-digit percent**;
- 2.5 GB/s is not a half-expert read shape either. Half-expert reads measure **4.53–6.47** here. To sit
  at 2.5 the requests must be far smaller than that, or the limit is submission/effective queue depth
  rather than placement.

So repacking is still worth doing for other reasons, but not as a bandwidth lever.

## What this leaves

**The E27T is already close to saturated for expert-sized reads.** io_uring therefore cannot produce a
large bandwidth gain for full-expert loads; its value is at smaller request sizes and in reducing
submission overhead.

Together with the two results in #1 — NVFP4 is losslessly incompressible (order-0 nibble entropy
**3.969 of 4.000**, combined floor 2.8 %) and expert paging is already near its ceiling (Belady replay:
global LRU **90.1 %** against a **95.6 %** oracle, with every practical alternative worse) — that closes
the byte-level and layout routes together.

**The remaining wins have to come from fewer misses, fewer bytes per missed expert, or hiding misses
behind compute — not from making each large read more sequential.**

## Proofs

Commit-pinned, all verified resolving:

- read-size / queue-depth probes — [`nvme-readshape-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvme-readshape-probe.py.txt), [`nvme-qd-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvme-qd-probe.py.txt)
- NVFP4 entropy measurement — [`nvfp4ent.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvfp4ent.py.txt)
- Belady replay harness and raw output — [`beladyreplay.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/beladyreplay.py.txt), [`belady-replay.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/belady-replay.txt)
- write-ups, det-198 and det-199 — [`determinism-investigation.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/determinism-investigation.md)

## Caveats

Our drive, and our benchmark structure. Our QD1 18 MiB figure varied 3.43–5.06 across runs. The paging
replay is Qwen3.8-Flash-Next (512 experts top-10, 48 layers, calibration text, no prefill or session
boundaries) against your 384 top-6 over 40 layers on agent traffic — carry across the *gaps between
policies*, not the absolute hit rates. The NVFP4 entropy result is about the format itself and is the
most portable of the three.

Drafted with AI assistance; every number re-derived, and the two withdrawn claims above were caught by
checking the published conclusions against the published table.
