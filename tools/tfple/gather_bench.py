#!/usr/bin/env python3
"""T1 (TensorFold EXL3 n-gram gather): 0.6.0's single-thread byte-offset gather vs whole-row views on threads.
Real EXL3 pack, random row ids (n-gram ids are hashes, so uniform is the realistic access), id counts for a decode
round (7 tokens x 16 heads = 112), a 4-stream round (448), an 8-stream round (896, T2 only) and prompt chunks (2048 / 8192 tokens x 16 = 32768 / 131072).
cold: the table file's pages dropped (POSIX_FADV_DONTNEED) before every rep; warm: read once first. Bytes compared.
Prints ONE json. argv: <tensorfold src dir> <model dir> [cold|warm|both] [reps]"""
import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

src, model = sys.argv[1], Path(sys.argv[2])
modes = ["cold", "warm"] if (sys.argv[3] if len(sys.argv) > 3 else "both") == "both" else [sys.argv[3]]
reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
sys.path.insert(0, src)
from tensorfold.families.qwen4_exp.cuda import exl3_pack                      # noqa: E402

FILE = model / "ngram_embedding.safetensors"
BASE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
pk = exl3_pack.Pack(model)
table = exl3_pack.NgramTable(pk, BASE, 128, "cpu")


def old_gather(t, ids):
    """0.6.0's NgramTable.gather (c464617), byte maps rebuilt here."""
    if not hasattr(t, "_old"):
        files, fidx, offsets = [], [], []
        for i in range(128):
            f, b, *_ = pk.entry(f"{BASE}shard_{i}.trellis")
            if f not in files:
                files.append(f)
            fidx.append(files.index(f)); offsets.append(b)
        t._old = ([np.memmap(model / f, dtype=np.uint8, mode="r") for f in files],
                  np.array(fidx, dtype=np.int64), np.array(offsets, dtype=np.int64))
    maps, fidx, offsets = t._old
    flat = np.asarray(ids, dtype=np.int64).reshape(-1)
    shard = np.searchsorted(t.starts, flat, side="right") - 1
    at = offsets[shard] + (flat - t.starts[shard]) * t.row_bytes
    where = fidx[shard]
    out = np.empty((len(flat), t.row_bytes), dtype=np.uint8)
    cols = np.arange(t.row_bytes)
    for f in np.unique(where):
        sel = np.nonzero(where == f)[0]
        out[sel] = maps[f][at[sel, None] + cols]
    return out.view(np.int16)


def drop():
    fd = os.open(FILE, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)


res = {"rows": table.rows, "row_bytes": table.row_bytes, "threads": exl3_pack.__dict__.get("GATHER_THREADS"),
       "src": src, "cells": []}
rng = np.random.default_rng(7)
for mode in modes:
    for n in (112, 448, 32768, 131072):
        ids = [rng.integers(0, table.rows, n) for _ in range(reps)]
        cell = {"mode": mode, "ids": n}
        for name, fn in (("old", lambda x: old_gather(table, x)), ("new", table.gather)):
            ts = []
            for r in range(reps):
                if mode == "cold":
                    drop()
                else:
                    fn(ids[r])
                t0 = time.perf_counter(); out = fn(ids[r]); ts.append(time.perf_counter() - t0)
            cell[name + "_ms"] = round(sorted(ts)[len(ts) // 2] * 1e3, 3)
            cell[name + "_all"] = [round(t * 1e3, 2) for t in ts]
        cell["equal"] = bool(np.array_equal(old_gather(table, ids[0]), table.gather(ids[0])))
        res["cells"].append(cell)
    # T2: decode rounds (c=1, c=4, c=8 under --parallel) with the cold fill off, adaptive, and forced on
    cf = getattr(table, "_cold", None)
    for n in (112, 448, 896) if cf is not None else ():
        ids = [rng.integers(0, table.rows, n) for _ in range(reps)]
        cell = {"mode": mode, "ids": n, "t2": True}
        for name, ok, force in (("nofill", False, False), ("adaptive", True, False), ("forced", True, True)):
            ts = []
            for r in range(reps):
                if mode == "cold":
                    drop(); cf.cold = True
                else:
                    cf.ok = True; table.gather(ids[r]); table.gather(ids[r])           # warm and settled
                cf.ok = ok
                if force:
                    cf.cold = True
                t0 = time.perf_counter(); table.gather(ids[r]); ts.append(time.perf_counter() - t0)
            cell[name + "_ms"] = round(sorted(ts)[len(ts) // 2] * 1e3, 3)
            cell[name + "_all"] = [round(t * 1e3, 2) for t in ts]
        cf.ok = True
        res["cells"].append(cell)
print(json.dumps(res))
