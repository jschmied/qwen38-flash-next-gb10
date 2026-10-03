#!/usr/bin/env python3
"""Time the n-gram gather of an 8K Flash Next prompt on the loaded engine: the table's gather as is, its raw rows,
the LUT step alone, and the LUT on threads (bytes compared). argv: <tensorfold src> <model dir>"""
import sys, sysconfig, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np                                                               # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402
from tensorfold.families.qwen4_exp import host_table as ht                      # noqa: E402

model = Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))[:8202]
eng = FlashNextEngine(model, max_len=16384)
w = eng.e.w if hasattr(eng, "e") and eng.e is not None else eng.w
layers = [l for l in w.layers if l.ple is not None]
print("PLE layers:", len(layers), "table", type(layers[0].ple.table).__name__)
for li, layer in enumerate(layers[:2]):
    p = layer.ple
    rows = p.ngram.ids(np.zeros((0,), dtype=np.int64), np.asarray(ids, dtype=np.int64))
    print("ids shape", rows.shape, "row width", getattr(p.table, "width", None))
    for rep in range(3):
        t0 = time.perf_counter(); full = p.table.gather(rows); t1 = time.perf_counter()
        if isinstance(p.table, ht.FP8Table):
            raw = ht.BF16Table.gather(p.table, rows); t2 = time.perf_counter()
            lut = p.table.lut[raw]; t3 = time.perf_counter()
            pool = ThreadPoolExecutor(16)
            out = np.empty(raw.shape, dtype=np.uint16)
            parts = np.array_split(np.arange(raw.shape[0]), 16)
            def conv(at):
                out[at] = p.table.lut[raw[at]]
            list(pool.map(conv, parts)); t4 = time.perf_counter()
            print(f"layer {li} rep {rep}: gather {1e3*(t1-t0):.1f} ms | raw rows {1e3*(t2-t1):.1f} | LUT {1e3*(t3-t2):.1f} | "
                  f"LUT on 16 threads {1e3*(t4-t3):.1f} | equal {np.array_equal(out, full) and np.array_equal(lut, full)} | "
                  f"bytes raw {raw.nbytes/1e6:.1f} MB out {full.nbytes/1e6:.1f} MB")
        else:
            print(f"layer {li} rep {rep}: gather {1e3*(t1-t0):.1f} ms")
