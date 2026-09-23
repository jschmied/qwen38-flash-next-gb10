#!/usr/bin/env python3
"""Finding 225 part 2: CPU prefetch before the GPU gather, fresh (never-touched) row ids per arm."""
import os, sys, time, json, ctypes, mmap
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/opt/llm/runners")
from pageable_tensor import map_file_as_cuda
PATH = sys.argv[1]
dev, fd, host = map_file_as_cuda(PATH, 160)
ROWS = dev.shape[0]; hostv = host.reshape(ROWS, 160)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
base = host.__array_interface__["data"][0]
if os.environ.get("RANDOM_ADV", "1") == "1":
    assert libc.madvise(ctypes.c_void_p(base), ctypes.c_size_t(host.size), 1) == 0  # MADV_RANDOM: no readahead
rng = np.random.default_rng(int(os.environ.get("SEED", "777")))
used = np.zeros(ROWS // 25 + 1, dtype=bool)   # 4096/160 ≈ 25.6 rows per page; track by page
def fresh(n):
    out = []
    while len(out) < n:
        c = rng.integers(0, ROWS, n * 2)
        pg = (c * 160) // 4096
        ok = ~used[np.minimum(pg, used.size - 1)]
        c = c[ok]; used[np.minimum((c * 160) // 4096, used.size - 1)] = True
        out.extend(c.tolist())
    return np.array(out[:n], dtype=np.int64)

def gpu_gather(ids):
    t = torch.from_numpy(ids).cuda(); torch.cuda.synchronize()
    t0 = time.perf_counter(); torch.index_select(dev, 0, t); torch.cuda.synchronize()
    return time.perf_counter() - t0

def pages_of(ids):
    b = ids * 160
    return np.unique(np.concatenate([b // 4096, (b + 159) // 4096]))

def pf_fadvise(ids):
    for p in pages_of(ids): os.posix_fadvise(fd, int(p) * 4096, 4096, os.POSIX_FADV_WILLNEED)
def pf_madvise(ids):
    for p in pages_of(ids): libc.madvise(ctypes.c_void_p(base + int(p) * 4096), ctypes.c_size_t(4096), 3)  # MADV_WILLNEED
POOL = ThreadPoolExecutor(32)
def pf_touch32(ids):
    chunks = np.array_split(np.sort(ids), 32)
    list(POOL.map(lambda c: int(hostv[c, 0].sum()), chunks))
def pf_touch1(ids): int(hostv[np.sort(ids), 0].sum())
def pf_both32(ids):
    chunks = np.array_split(np.sort(ids), 32)
    list(POOL.map(lambda c: int(hostv[c, 0].sum()) + int(hostv[c, 159].sum()), chunks))
POOL64 = ThreadPoolExecutor(64)
def pf_both64(ids):
    chunks = np.array_split(np.sort(ids), 64)
    list(POOL64.map(lambda c: int(hostv[c, 0].sum()) + int(hostv[c, 159].sum()), chunks))

res = {}
ARM = os.environ["ARM"]; SIZES = [int(x) for x in os.environ.get("SIZES", "64,480000").split(",")]
for n, reps in ((n, 20 if n < 1000 else 1) for n in SIZES):
    for name, fn in (("none", None), ("touch32", pf_touch32), ("both32", pf_both32), ("both64", pf_both64), ("fadvise", pf_fadvise)):
        if name != ARM: continue
        tp, tg = [], []
        for _ in range(reps):
            ids = fresh(n)
            t0 = time.perf_counter()
            if fn: fn(ids)
            tp.append(time.perf_counter() - t0); tg.append(gpu_gather(ids))
        tot = [a + b for a, b in zip(tp, tg)]
        res[f"n{n}_{name}"] = {"prefetch_ms_med": round(1e3 * float(np.median(tp)), 3),
                               "gather_ms_med": round(1e3 * float(np.median(tg)), 3),
                               "total_ms": [round(1e3 * min(tot), 3), round(1e3 * float(np.median(tot)), 3), round(1e3 * max(tot), 3)]}
        print(f"n{n}_{name}", json.dumps(res[f"n{n}_{name}"]), flush=True)
