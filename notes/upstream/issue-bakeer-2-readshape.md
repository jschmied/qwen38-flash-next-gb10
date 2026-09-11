Follow-up to #1, and the more useful half of it. Measured on a GB10 box's NVMe with `O_DIRECT` on real
safetensors shard bytes, so the page cache cannot flatter the numbers.

**The short version: read *size* and queue depth substitute for each other, and at expert size it is
size that matters. On this curve your reported ~2.5 GB/s is where ~1 MiB at QD1 lands — an order of
magnitude below one 18.8 MB expert.**

## The curve

| read size | QD1 GB/s | QD4 GB/s | depth gain |
| --- | --- | --- | --- |
| 64 KiB | 0.46 | 0.99 | 2.2× |
| 256 KiB | 1.07 | 2.57 | 2.4× |
| 1 MiB | 2.40 | 5.70 | **2.4×** |
| 4 MiB | 4.02 | 6.48 | 1.6× |
| 9 MiB — half an expert | 4.53 | 6.47 | 1.4× |
| 18 MiB — one expert | 5.06 | 6.42 | **1.27×** |

Queue depth alone saturates at **QD2** for expert-sized reads: 5.06 → 6.39, then flat through QD32.

So "add io_uring" is the wrong framing of this lever, and it was mine in #1. The right one is *make
each read big; pay for depth only when you cannot.*

## Why this points at your layout

`LIMITATIONS.md` says the NVMe delivers ~2.5 GB/s "at the read sizes and queue depths a decode step
produces", against a ceiling nearer 5.5. On the curve above, 2.5 GB/s is **~1 MiB at QD1** or
**~256 KiB at QD4**. Each expert is 18.8 MB, and your packing stores it as a scale-run plus a separate
weight-run at a different offset — which is exactly the shape that would land there.

If that is what is happening, **expert-major repacking is worth up to ~2.2×** on the streaming path.
That is already your own item; this is evidence for promoting it, not a new idea.

## Which matters because the byte-level routes are closed

We tested the two obvious ways to move fewer bytes, and both are dead:

| route | result |
| --- | --- |
| lossless compression of NVFP4 | **2.8 % floor.** Order-0 nibble entropy **3.969 of 4.000** over 73.7 M nibbles; order-1 conditional 3.968. Per-group amax scaling exists to make each group use the full code range, so near-uniform occupancy is evidence the quantizer works. Expect the same of CB3. |
| smarter expert paging | **LRU wins.** Belady replay on real (layer, expert) sequences, 7.2 M scored accesses, 25.6 % resident: static trace-ranked 55.4 %, **global LRU 90.1 %**, decayed LFU 88.5 %, protected-set + admission 84.8 %, Belady oracle 95.6 %. A frequency-ranked protected segment is 5.3 pp *worse* than plain LRU. The 5.5 pp oracle gap is only 10.9 % reachable with a 6-token lookahead, so speculation does not act as a partial oracle. |

With the bytes fixed and the cache near-optimal, **read shape is the remaining lossless lever**, and on
these measurements it is the biggest one by a wide margin.

## Proofs

Commit-pinned, all verified resolving:

- read-size / queue-depth probe — [`nvme-readshape-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvme-readshape-probe.py.txt), [`nvme-qd-probe.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvme-qd-probe.py.txt)
- NVFP4 entropy measurement — [`nvfp4ent.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/nvfp4ent.py.txt)
- Belady replay harness and raw output — [`beladyreplay.py.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/beladyreplay.py.txt), [`belady-replay.txt`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/data/belady-replay.txt)
- write-ups, det-198 and det-199 — [`determinism-investigation.md`](https://github.com/jschmied/qwen38-flash-next-gb10/blob/f8aa6906251a69b4e08889caa41d40b62ae2ca2e/notes/determinism-investigation.md)

## Caveats

Our drive, not yours — model, firmware and fill state all move the absolute GB/s, and our QD1 18 MiB
figure varied 3.43–5.06 across runs. **The shape of the curve is what transfers, not the numbers.**
The paging replay is Qwen3.8-Flash-Next (512 experts top-10, 48 layers, calibration text, no prefill or
session boundaries); yours is 384 top-6 over 40 layers on agent traffic, so I would carry across the
*gaps between policies* and not the absolute hit rates. The entropy result is about NVFP4 itself and is
the most portable of the three.

Drafted with AI assistance; every number re-derived and every link checked before posting.
