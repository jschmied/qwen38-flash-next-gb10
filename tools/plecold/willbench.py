"""Cold-page fill without compiled code (follow-up PR to #58439): MADV_WILLNEED readahead + touch, single thread,
vs the C helper (16 pthreads, MADV_POPULATE_READ). 56 random rows of a PLE shard per rep = ~57 cold pages; page cache
dropped once before the run, fresh random rows each rep. Run as root on an idle box (no server, no SSD traffic)."""
import ctypes, mmap, os, statistics as st, time
import numpy as np

F = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-00003.safetensors"
SZ = os.path.getsize(F); RB = 160
fd = os.open(F, os.O_RDONLY)
m = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); m.madvise(mmap.MADV_RANDOM)
arr = np.frombuffer(m, dtype=np.uint8); rows = (SZ - 8192) // RB
base = arr.__array_interface__["data"][0]
rng = np.random.default_rng(1)
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]


def pages(r):
    a = 8192 + r * RB; b = a + RB - 1
    return np.unique(np.concatenate([a & ~4095, b & ~4095]))          # offsets into the mapping


def resident(offs):
    vec = (ctypes.c_ubyte * 1)()
    return sum(1 for o in offs if libc.mincore(ctypes.c_void_p(base + int(o)), 4096, vec) == 0 and vec[0] & 1)


def willneed_touch(offs):
    for o in offs:
        m.madvise(mmap.MADV_WILLNEED, int(o), 4096)                   # async readahead, returns at once
    s = 0
    for o in offs:
        s += int(arr[int(o)])                                              # waits for the page if still in flight
    return s


libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def willneed_then_populate(offs):
    for o in offs:
        libc.madvise(ctypes.c_void_p(base + int(o)), 4096, 3)        # MADV_WILLNEED via ctypes (GIL released)
    for o in offs:
        libc.madvise(ctypes.c_void_p(base + int(o)), 4096, 22)       # MADV_POPULATE_READ: waits, GIL released


def willneed_only(offs):
    for o in offs:
        m.madvise(mmap.MADV_WILLNEED, int(o), 4096)


lib = ctypes.CDLL("/opt/llm/runners/plecold/libfnpopulate.so")
lib.fn_populate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def c_helper(offs):
    pg = (offs + base).astype(np.uint64)
    return lib.fn_populate(pg.ctypes.data, len(pg), 16)


def serial_touch(offs):
    s = 0
    for o in offs:
        s += int(arr[int(o)])
    return s


res = {}
for name, fn in (("serial_touch", serial_touch), ("c_helper_16thr", c_helper),
                 ("willneed_then_touch", willneed_touch), ("willneed_then_populate", willneed_then_populate), ("willneed_only(issue)", willneed_only)):
    ts, cold = [], []
    for rep in range(25):
        offs = pages(np.sort(rng.integers(0, rows, 56)))
        cold.append(len(offs) - resident(offs))
        t = time.perf_counter(); fn(offs); ts.append((time.perf_counter() - t) * 1000)
    res[name] = {"median_ms": round(st.median(ts), 3), "p90_ms": round(sorted(ts)[int(0.9 * len(ts))], 3),
                 "cold_pages_median": st.median(cold)}
    print(name, res[name], flush=True)
    if name == "willneed_only(issue)":
        time.sleep(0.05)
        print("  resident after 50 ms of the last issue:", resident(offs), "/", len(offs), flush=True)
import json
json.dump(res, open("/opt/llm/runners/results/willbench.json", "w"), indent=1)
print("== ALL DONE ==")
