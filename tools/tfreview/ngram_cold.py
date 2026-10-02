#!/usr/bin/env python3
"""#252 review: in-process prefill of a 24,576-token prompt with the n-gram table cold (process mappings dropped with
MADV_DONTNEED, the file's page cache evicted with POSIX_FADV_DONTNEED) and warm. Records prefill time, the first 16
tokens' hash, major faults (whole process) and how much of the table file is resident afterwards (mincore).
TF_NGRAM_AHEAD is read by the tree at import (set it in the environment). NGRAM_RANDOM=1 also sets MADV_RANDOM on the
table mappings (no fault read-around) to see whether the residency after a cold run is read-around or WILLNEED.
argv: <tensorfold src> <model dir> <label>"""
import ctypes, hashlib, json, mmap, os, resource, sys, sysconfig, time
from pathlib import Path

src, model, label = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, src)
import numpy as np                                                               # noqa: E402
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

libc = ctypes.CDLL(None, use_errno=True)
libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
libc.munlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
PG = mmap.PAGESIZE

tok = AutoTokenizer.from_pretrained(model)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[:600]))
eng = FlashNextEngine(model, max_len=32768)
eng.generate(tok.encode("# warm\n") + ids[:1024], 1, None, lambda t: None)
torch.cuda.synchronize()
t = next(lay.ple for lay in eng.w.layers if lay.ple is not None).table
arrays = list(t.maps) + [a for k in ("words", "scales", "biases") for a in getattr(t, k, [])]
# the engine mlocks the table when its startup budget has room (3.05 on GB10 does); the PR's cold case is the table
# NOT locked (a pack whose table does not fit, or pressure). Unlock it so eviction can take; the startup line says
# whether it was locked.
unlocked = sum(libc.munlock(a.ctypes.data, a.nbytes) == 0 for a in arrays)
paths = sorted({Path(a.filename) for a in t.maps})
size = sum(p.stat().st_size for p in paths)
if os.environ.get("NGRAM_RANDOM") == "1":
    for a in t.maps:
        libc.madvise(a.ctypes.data, a.nbytes, mmap.MADV_RANDOM)


def resident() -> int:
    """Resident bytes of the table file(s), through a fresh mapping of each (mincore)."""
    n = 0
    for p in paths:
        with open(p, "rb") as f:
            m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            a = np.frombuffer(m, dtype=np.uint8)
            vec = np.zeros((len(m) + PG - 1) // PG, dtype=np.uint8)
            libc.mincore(ctypes.c_void_p(a.ctypes.data), len(m), vec.ctypes.data)
            n += int((vec & 1).sum()) * PG
            del a
            m.close()
    return n


def cold() -> None:
    for a in arrays:
        base = a.ctypes.data & ~(PG - 1)
        libc.madvise(base, a.ctypes.data + a.nbytes - base, mmap.MADV_DONTNEED)
    for p in paths:
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


for i, state in enumerate(("cold", "cold", "warm")):
    # a new prompt each run (the second run of one prompt is a prefix-cache hit); the same three in every arm
    p = tok.encode(f"# prompt {i} of 24576 tokens\n") + ids[i * 24576:(i + 1) * 24576]
    if state == "cold":
        cold()
    else:
        t.prefetch()
    r0, f0, c0 = resident(), resource.getrusage(resource.RUSAGE_SELF).ru_majflt, time.perf_counter()
    got = []
    st = eng.generate(p, 16, None, lambda tt: got.extend(tt) and None, stop_eos=False)
    torch.cuda.synchronize()
    time.sleep(2)                                                    # let a background read-ahead finish
    print(json.dumps({"label": label, "state": state, "ahead": os.environ.get("TF_NGRAM_AHEAD", "1"),
                      "random": os.environ.get("NGRAM_RANDOM", "0"), "tokens": len(p), "run": i, "unlocked": unlocked,
                      "prefill_s": round(st.get("prefill_s", -1), 3), "wall_s": round(time.perf_counter() - c0 - 2, 3),
                      "majflt": resource.getrusage(resource.RUSAGE_SELF).ru_majflt - f0,
                      "table_gib": round(size / 2**30, 2), "resident_before_gib": round(r0 / 2**30, 2),
                      "resident_after_gib": round(resident() / 2**30, 2),
                      "hash": hashlib.sha256(json.dumps(got).encode()).hexdigest()[:12]}), flush=True)
