"""Offline CPU-prefetch microbench (finding 226 debug): v1 touch vs v2 touch variants, cold page cache per arm.
HYPOTHESIS: arm v2cur reproduces the server's 3-9x slowdown vs v1 (v1 ~50-170 ms for 51,200 rows);
v2sorted (offsets sorted per file, 64 even chunks) is within 1.3x of v1.
IF OUTSIDE: v2cur ~= v1 offline means the slowdown is GPU-fault/CPU-touch interaction, not the touch code."""
import os, sys, time, json, glob, struct, mmap, ctypes, numpy as np
from concurrent.futures import ThreadPoolExecutor
ARM = sys.argv[1]; SEED = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 51200
libc = ctypes.CDLL("libc.so.6", use_errno=True); POOL = ThreadPoolExecutor(64)
def mapf(f):
    fd = os.open(f, os.O_RDONLY); sz = os.path.getsize(f)
    mm = mmap.mmap(fd, sz, mmap.MAP_SHARED, mmap.PROT_READ); a = np.frombuffer(mm, np.uint8)
    libc.madvise(ctypes.c_void_p(a.__array_interface__["data"][0]), ctypes.c_size_t(sz), 1)
    return mm, a
rng = np.random.default_rng(SEED); rows = rng.integers(0, 320001536, N)
if ARM == "v1":
    _, a = mapf("/opt/llm/models/ple-cache/qwen38-fn-plefp8.bin"); h = a.reshape(-1, 160)
    t0 = time.perf_counter()
    r = np.sort(rows); list(POOL.map(lambda c: int(h[c, 0].sum()) + int(h[c, -1].sum()), np.array_split(r, 64)))
else:
    d = "/opt/llm/models/qwen38-flash-next-mtpfp4"; shards = {}; files = {}
    for f in sorted(glob.glob(d + "/model-plefp8-*.safetensors")):
        with open(f, "rb") as b:
            n = struct.unpack("<Q", b.read(8))[0]; hd = json.loads(b.read(n))
        for k, v in hd.items():
            if ".shard_" in k and k.endswith(".weight"):
                shards[int(k.split(".shard_")[1].split(".")[0])] = (f, 8 + n + v["data_offsets"][0])
        files[f] = mapf(f)[1]
    fl = list(files); S = 2500012
    sf = np.array([fl.index(shards[i][0]) for i in range(128)]); so = np.array([shards[i][1] for i in range(128)])
    arrs = [files[f] for f in fl]
    t0 = time.perf_counter()
    s = rows // S; fidx = sf[s]; off = so[s] + (rows - s * S) * 160
    if ARM == "v2cur":
        order = np.argsort(fidx, kind="stable"); fidx, off = fidx[order], off[order]
        cuts = np.flatnonzero(np.diff(fidx)) + 1
        tasks = []
        for g, o in zip(np.split(fidx, cuts), np.split(off, cuts)):
            for c in np.array_split(o, max(1, min(64, o.size // 256))): tasks.append((arrs[int(g[0])], c))
    elif ARM == "v2rows":   # per-shard 2-D views (rows x 160), sorted rows, 64 chunks: same indexing as v1
        views = [arrs[sf[i]][so[i]:so[i] + S * 160].reshape(S, 160) for i in range(128)]
        r = np.sort(rows); sh = r // S; loc = r - sh * S
        tasks = []
        for ci in np.array_split(np.arange(r.size), 64):
            if not ci.size: continue
            cs = sh[ci]; cuts = np.flatnonzero(np.diff(cs)) + 1
            for g, l in zip(np.split(cs, cuts), np.split(loc[ci], cuts)): tasks.append((views[int(g[0])], l))
        list(POOL.map(lambda t: int(t[0][t[1], 0].sum()) + int(t[0][t[1], -1].sum()), tasks))
        tasks = []
    elif ARM == "v2flatsame":   # ISOLATION: exactly v2rows' sorted rows and task split, but flat byte indexing
        # HYPOTHESIS (review 2): if ordering/grouping is the cause, this lands within 1.3x of v2rows;
        # if >1.5x, the 2-D indexing path itself contributes.
        r = np.sort(rows); sh = r // S; loc = r - sh * S
        tasks = []
        for ci in np.array_split(np.arange(r.size), 64):
            if not ci.size: continue
            cs = sh[ci]; cuts = np.flatnonzero(np.diff(cs)) + 1
            for g, l in zip(np.split(cs, cuts), np.split(loc[ci], cuts)):
                i = int(g[0]); tasks.append((arrs[sf[i]], so[i] + l * 160))
        list(POOL.map(lambda t: int(t[0][t[1]].sum()) + int(t[0][t[1] + 159].sum()), tasks))
        tasks = []
    elif ARM.startswith("flatk"):   # review 3: NumPy GIL threshold test, flat 1-D indexing with K indices per task
        K = int(ARM[5:]); r = np.sort(rows); sh = r // S; loc = r - sh * S
        fi = sf[sh]; off = so[sh] + loc * 160
        order = np.lexsort((off, fi)); fi, off = fi[order], off[order]
        tasks = []
        for f in np.unique(fi):
            o = off[fi == f]
            for c in range(0, o.size, K): tasks.append((arrs[int(f)], o[c:c + K]))
        list(POOL.map(lambda t: int(t[0][t[1]].sum()) + int(t[0][t[1] + 159].sum()), tasks))
        tasks = []
    else:  # v2sorted
        key = fidx * (1 << 40) + off; order = np.argsort(key); fidx, off = fidx[order], off[order]
        tasks = []
        for ci in np.array_split(np.arange(fidx.size), 64):
            if not ci.size: continue
            for fi in np.unique(fidx[ci]):
                m = ci[fidx[ci] == fi]; tasks.append((arrs[int(fi)], off[m]))
    list(POOL.map(lambda t: int(t[0][t[1]].sum()) + int(t[0][t[1] + 159].sum()), tasks))
print(json.dumps({"arm": ARM, "rows": N, "ms": round((time.perf_counter() - t0) * 1e3, 1)}))
