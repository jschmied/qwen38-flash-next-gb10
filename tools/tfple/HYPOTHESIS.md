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
