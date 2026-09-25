import os, mmap, time, numpy as np, statistics as st
from concurrent.futures import ThreadPoolExecutor
F = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-00003.safetensors"
SZ = os.path.getsize(F); RB = 160
fd = os.open(F, os.O_RDONLY); m = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); os.close(fd); m.madvise(mmap.MADV_RANDOM)
arr = np.frombuffer(m, dtype=np.uint8); rows = (SZ - 8192) // RB; view = arr[8192:8192 + rows * RB].reshape(rows, RB)
pool = ThreadPoolExecutor(64); rng = np.random.default_rng(0)
def fault(ix): return int(view[ix, 0].sum()) + int(view[ix, RB - 1].sum())
list(pool.map(fault, [np.array([1])] * 64))           # warm the pool
for n in (56,):
    for k in (1, 4, 8, 16, 32, 56):
        ts = []
        for rep in range(15):
            r = np.sort(rng.integers(0, rows, n))      # fresh random rows: cold pages
            t = time.perf_counter()
            if k == 1: fault(r)
            else: list(pool.map(fault, np.array_split(r, k)))
            ts.append((time.perf_counter() - t) * 1000)
        print(f"rows={n} tasks={k:2d}: median {st.median(ts):6.2f} ms  min {min(ts):5.2f}", flush=True)
import ctypes
libc = ctypes.CDLL("libc.so.6", use_errno=True); libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
base = arr.__array_interface__["data"][0]
def pages(r):
    a = base + 8192 + r * RB; b = a + RB - 1
    return np.unique(np.concatenate([a & ~4095, b & ~4095]))
def pop(pg):
    for p in pg: libc.madvise(ctypes.c_void_p(int(p)), 4096, 22)       # MADV_POPULATE_READ, GIL released
for k in (1, 8, 16, 32, 56):
    ts = []
    for rep in range(15):
        pg = pages(np.sort(rng.integers(0, rows, 56)))
        t = time.perf_counter()
        if k == 1: pop(pg)
        else: list(pool.map(pop, np.array_split(pg, k)))
        ts.append((time.perf_counter() - t) * 1000)
    print(f"POPULATE_READ pages={len(pg)} tasks={k:2d}: median {st.median(ts):6.2f} ms  min {min(ts):5.2f}", flush=True)
lib = ctypes.CDLL("/opt/llm/runners/plecold/libfnpopulate.so"); lib.fn_populate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
for thr in (8, 16, 32, 57):
    ts = []
    for rep in range(15):
        pg = pages(np.sort(rng.integers(0, rows, 56))).astype(np.uint64)
        t = time.perf_counter(); e = lib.fn_populate(pg.ctypes.data, len(pg), thr); ts.append((time.perf_counter() - t) * 1000)
    print(f"C fn_populate pages={len(pg)} threads={thr:2d}: median {st.median(ts):6.2f} ms  min {min(ts):5.2f} errors {e}", flush=True)
