#!/usr/bin/env python3
"""Fresh-row n-gram gathers (a new 8K prompt each) before and after MADV_POPULATE_READ on the table's mappings, and the
time and major/minor faults of each. argv: <tensorfold src> <model dir>"""
import ctypes, resource, sys, sysconfig, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import numpy as np                                                               # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

model = Path(sys.argv[2])
tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:900]))
eng = FlashNextEngine(model, max_len=16384)
w = eng.e.w if getattr(eng, "e", None) is not None else eng.w
p = next(l.ple for l in w.layers if l.ple is not None)
libc = ctypes.CDLL(None, use_errno=True)
libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)


def fresh(k):
    toks = np.asarray(ids[k * 8202:(k + 1) * 8202], dtype=np.int64)
    rows = p.ngram.ids(np.zeros((0,), dtype=np.int64), toks)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter(); p.table.gather(rows); t1 = time.perf_counter()
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return round(1e3 * (t1 - t0), 1), r1.ru_minflt - r0.ru_minflt, r1.ru_majflt - r0.ru_majflt


for k in (1, 2, 3):
    print("before populate: prompt", k, "ms / minor / major faults", fresh(k), flush=True)
t0 = time.perf_counter()
rc = [libc.madvise(a.ctypes.data, a.nbytes, 22) for a in p.table.values]           # MADV_POPULATE_READ
print(f"populate {sum(a.nbytes for a in p.table.values) / 2**30:.1f} GiB: {time.perf_counter() - t0:.1f} s, rc {set(rc)}",
      flush=True)
for k in (4, 5, 6):
    print("after populate: prompt", k, "ms / minor / major faults", fresh(k), flush=True)
