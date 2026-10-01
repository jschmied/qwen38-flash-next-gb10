# TensorFold n-gram table work (T1–T4), written before each measurement (2026-10-01)

Branch `ple-gather` in the worktree `~/git/tensorfold-ple` (base 0.6.0 `c464617`). Real pack:
`qwen38-flash-next-exl3-3.05bpw`, 128 shards × 2,500,012 rows × 102 B (5-bit rows), 16 ids per token.

## T1 — EXL3 gather: whole-row views on threads vs 0.6.0's single-thread byte offsets (`gather_bench.py`)
Our finding 226 (vLLM, BF16 table): flat byte-offset indexing 12× slower than 2-D row views for 51,200 cold rows;
finding 225: threads 32 → 64 took 480k rows 2.8 → 1.6 s.
- Prompt chunks (32,768 / 131,072 ids), cold: **5–15× faster** (16 threads, row views); warm: **2–5×**.
- Decode rounds (112 / 448 ids, below the 1,024-id thread cut): row view only, **1–3×** warm; cold dominated by the
  page faults (~60 µs each, serial) and roughly **equal**.
- Bytes identical in every cell. Out of range → profile before changing anything else.

## T2 — adaptive cold fill for decode-sized gathers (same `gather_bench.py`, T2 cells)
Ours: §4x/§4z, vllm#58835: 57 cold pages 4.6 → 0.36 ms.
- Cold (pages dropped), 112 / 448 / 896 ids: fill **4–12× faster** than no fill (the serial faults become one batch).
- Warm, adaptive (settled warm, so no madvise): within **±0.05 ms** of no fill (two `getrusage` calls).
- Warm, forced fill (the worst case if the guess were wrong): **+0.2…+2 ms** at 112…896 ids (two calls a page),
  which is why it is adaptive.

## T4 — why 0.6.0's EXL3 start took 22.2 s to lock the table (0.5.0: 0.3–0.4 s in six starts)
Two candidates: (a) `Pack.release()` drops the table file's pages (whole-file `POSIX_FADV_DONTNEED`) — but it only
covers pages read before layer 1 finishes, so it should cost little; (b) memory pressure: 0.6.0 ran at
`--context 262144 --parallel 8` (estimate 61.97 GiB, 27 GiB free at ready), and the page cache gave up the table
between the prefetch and the lock. Test: two stock 0.6.0 EXL3 starts with the table file's residency sampled every
2 s (`resident.py`), at 262144/8 and at 65536/4 (the 0.5.0 runs' shape).
- Expected: **(b)**: residency reaches ~100 % during the prefetch, then falls before the lock at 262144/8, and the lock
  takes > 10 s there; at 65536/4 it stays resident and locks in < 1 s. If residency falls right after layer 1 in both,
  it is (a).

## Re-measure (gather_bench2.py, one fresh process per cell, own ids) — T1 v2 (one row view per file) + T2
The first run's cold cells were void (§5ao: DONTNEED cannot evict pages the process still maps).
- Warm, decode sizes (112/448/896): T1 v2 **≤ old** (one index per file again, row views); prompt sizes 1.8–2.5×
  faster (as §5ao).
- Cold, decode sizes: t1 ≈ old (serial faults either way, ~60 µs a page); **t1t2 4–12× faster** than both.
- Cold, prompt sizes: t1/t1t2 **5–15× faster** than old (16 threads overlapping their faults).
- Bytes equal in every cell.
