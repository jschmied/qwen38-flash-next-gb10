#!/usr/bin/env python3
"""T1/T2 re-measure (2026-10-01), replacing gather_bench.py's void cold cells: POSIX_FADV_DONTNEED cannot evict pages
a process still maps, so a cold cell must run in a FRESH process with its own ids. One call = one variant, one mode,
one id count, one set of ids (seeded); prints one json line.
variants: old (0.6.0's byte-offset gather, rebuilt here), t1 (branch gather, cold fill off), t1t2 (branch, adaptive
cold fill, starting cold as after a start). cold: the table file dropped from the page cache, then ONE timed gather;
warm: the same ids gathered once untimed, then timed.
argv: <branch src> <model dir> <variant> <cold|warm> <ids> <seed>"""
import json, os, sys, time
from pathlib import Path

import numpy as np

src, model, variant, mode, n, seed = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
sys.path.insert(0, src)
from tensorfold.families.qwen4_exp.cuda import exl3_pack                      # noqa: E402

FILE = model / "ngram_embedding.safetensors"
BASE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
pk = exl3_pack.Pack(model)
table = exl3_pack.NgramTable(pk, BASE, 128, "cpu")
ids = np.random.default_rng(seed).integers(0, table.rows, n)

if variant == "old":
    files, fidx, offsets = [], [], []
    for i in range(128):
        f, b, *_ = pk.entry(f"{BASE}shard_{i}.trellis")
        if f not in files:
            files.append(f)
        fidx.append(files.index(f)); offsets.append(b)
    maps = [np.memmap(model / f, dtype=np.uint8, mode="r") for f in files]
    fidx, offsets = np.array(fidx, dtype=np.int64), np.array(offsets, dtype=np.int64)

    def gather(x):
        flat = np.asarray(x, dtype=np.int64).reshape(-1)
        shard = np.searchsorted(table.starts, flat, side="right") - 1
        at = offsets[shard] + (flat - table.starts[shard]) * table.row_bytes
        where = fidx[shard]
        out = np.empty((len(flat), table.row_bytes), dtype=np.uint8)
        cols = np.arange(table.row_bytes)
        for f in np.unique(where):
            sel = np.nonzero(where == f)[0]
            out[sel] = maps[f][at[sel, None] + cols]
        return out.view(np.int16)
else:
    table._cold.ok = variant == "t1t2"
    table._cold.cold = True
    gather = table.gather

if mode == "cold":
    fd = os.open(FILE, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
else:
    gather(ids)
    if variant == "t1t2":
        gather(ids)                                         # settles the adaptive gate warm
t0 = time.perf_counter(); out = gather(ids); dt = time.perf_counter() - t0
ref = np.stack([table.words[s][r] for s, r in zip(np.searchsorted(table.starts, ids, side="right") - 1,
                                                    ids - table.starts[np.searchsorted(table.starts, ids, side="right") - 1])])
print(json.dumps({"variant": variant, "mode": mode, "ids": n, "seed": seed, "ms": round(dt * 1e3, 3),
                  "equal": bool(np.array_equal(out, ref))}))
