> **Revision 2, 2026-09-11.** Revision 1 withdrew two wrong claims. This revision withdraws a third,
> in the *opposite* direction: I had concluded that read size dominates and io_uring could not help
> much. **That was an artefact of stopping my queue-depth sweep at QD4.** Swept properly, depth
> dominates and io_uring is worth up to 8×. The measurements below are all re-run; what changed is
> marked. I would rather revise this twice than leave a wrong curve standing.

Measured on a GX10 / GB10 box, `O_DIRECT` on real safetensors shard bytes so the page cache cannot
flatter anything. Drive: **Phison PS5027-E27T**, PCIe 4.0 ×4, DRAM-less with HMB
(`ESL01TBTLCZ-27J2-TYN`, 1 TB).

## 1. Queue depth dominates, not read size

| depth | 64 KiB GB/s | 256 KiB GB/s |
| --- | --- | --- |
| QD1 | 0.49 | 1.06 |
| QD4 — *where my first sweep stopped* | 1.05 | 2.35 |
| QD16 | 3.06 | 5.22 |
| QD64 | 3.87 | 5.84 |
| **QD128** | **3.90** | **5.88** |
| QD256 | 3.64 | 5.77 (thread overhead) |

**64 KiB improves 8.0× from QD1 to QD128. 256 KiB improves 5.5×, reaching 5.88 GB/s — 92 % of the
6.4 GB/s that 18 MiB reads achieve.**

So read size barely matters *given depth*. My earlier claim that "the E27T is close to saturated at
small reads" was wrong: at 64 KiB / QD1 I was measuring my own thread pool, not the drive. 0.46 GB/s
there is ~7,000 IOPS against a controller rated ~1.0–1.2 M 4K random-read IOPS — it was never
plausible as a hardware limit, and I should have caught that before publishing it.

**Consequence: io_uring is worth pursuing after all**, and finer-grained streaming is far more viable
than my first version implied — 256 KiB at depth costs 8 % against 18 MiB, not the 6.5× that a QD4
column suggests.

## 2. What still holds at expert size

| | |
| --- | --- |
| 18 MB contiguous through one file | 6.37 GB/s |
| 18 MB scattered across 206 files | 6.42 GB/s |

**Placement and seek locality cost nothing measurable at expert granularity.** And 6.4–6.5 is ~88 % of
Phison's 7.4 GB/s E27T rating, so at *that* size the drive really is close to saturated — provided
reads are issued concurrently. At QD1 you would get 5.06 instead of 6.42, a 21 % loss for free.

Also still withdrawn from revision 1: expert-major repacking is **not** worth ~2.2×. Combining
half-expert reads measures −0.8 % at QD4 and +11.7 % at QD1, and a real NVFP4 expert splits
asymmetrically (~17.7 MB weights + ~1.1 MB scales), so merging saves single-digit percent.

## 3. The slot is Gen5; the stock drive is Gen4

Prompted by a suggestion in the comments below, I checked the link topology:

```
root port  0004:00:00.0   LnkCap: Speed 32GT/s, Width x4     <- Gen5 capable
drive      0004:01:00.0   LnkCap: Speed 16GT/s, Width x4     <- Gen4, caps the link
                          LnkSta: Speed 16GT/s
```

**The M.2 slot negotiates PCIe 5.0 ×4; the stock Phison E27T is Gen4, so roughly half the available
link is unused.** Raw ceiling 7.88 → **15.75 GB/s** with a Gen5 drive.

For a streaming engine at ~0.9 GB per generated token:

| | ms/token | storage-only ceiling |
| --- | --- | --- |
| measured today, 6.45 GB/s | 140 | 7.2 tok/s |
| a Gen5 drive at ~13 GB/s | 69 | 14.4 tok/s |

That is the largest single lever I have found on the unpruned path, and it is a part swap rather than
a code change. Real gains depend on the replacement's behaviour at ~18 MB reads and on the rest of the
decode pipeline, neither of which I can measure without the hardware.

## 4. Why this matters more than byte-level work

The two byte-level routes are closed (details in #1):

- **NVFP4 is losslessly incompressible** — order-0 nibble entropy **3.969 of 4.000** over 73.7 M
  nibbles, combined floor **2.8 %**. Per-group amax scaling exists to make each group use the full code
  range, so near-uniform occupancy is evidence the quantizer works. Expect the same of CB3.
- **Expert paging is near its ceiling** — Belady replay, 7.2 M scored accesses, 25.6 % resident: static
  trace-ranked 55.4 %, **global LRU 90.1 %**, decayed LFU 88.5 %, protected-set + admission 84.8 %,
  oracle 95.6 %. Nothing beat plain LRU; a frequency-ranked protected segment was 5.3 pp worse. The
  5.5 pp oracle gap is only 10.9 % reachable with a 6-token lookahead.

So the remaining wins are **fewer misses, fewer bytes per miss, hiding misses behind compute — and
raw link bandwidth**, which section 3 says is sitting unused.

## Proofs

Commit-pinned at `556748b`, all verified resolving:

- depth sweep past QD4 — [`nvme-deepqd-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/nvme-deepqd-probe.py.txt)
- read-size / QD probes — [`nvme-readshape-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/nvme-readshape-probe.py.txt), [`nvme-qd-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/nvme-qd-probe.py.txt)
- NVFP4 entropy — [`nvfp4ent.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/nvfp4ent.py.txt)
- Belady replay + raw output — [`beladyreplay.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/beladyreplay.py.txt), [`belady-replay.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/data/belady-replay.txt)
- write-ups det-198 / 199 / 200 — [`determinism-investigation.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/556748b2f8ba3a203660d7ea5262a387847398bf/notes/determinism-investigation.md)

## Caveats

Our drive and our benchmark structure. Depth here came from a **thread pool, not io_uring** — which is
why QD256 regresses, and why true io_uring would likely do slightly better than the QD128 figures
rather than worse. Our QD1 18 MiB figure varied 3.43–5.06 across runs. Phison's 7.4 GB/s is a
controller capability, not a published rating for this OEM part (other E27T 1 TB references give 7.35;
some products 7.0). The paging replay is Qwen3.8-Flash-Next (512 experts top-10, 48 layers,
calibration text, no prefill or session boundaries) against your 384 top-6 on agent traffic — carry
across the gaps between policies, not the absolute hit rates.

Drafted with AI assistance. Three claims in earlier revisions were wrong; each was caught by
re-measuring rather than by re-reasoning, and each came from reading a conclusion off a table that
stopped too early.
