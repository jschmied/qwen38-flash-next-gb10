import os, mmap, random, time, numpy as np
from concurrent.futures import ThreadPoolExecutor
F = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-00005.safetensors"
SZ = os.path.getsize(F); PG = 4096
def drop():
    fd = os.open(F, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def offs(n, seed): r = random.Random(seed); return [r.randrange(SZ // PG) * PG for _ in range(n)]
# 1) O_DIRECT preads in threads (preadv releases the GIL): the SSD's parallel capacity
for T in (1, 8, 32, 64):
    drop(); o = offs(8000, T); fd = os.open(F, os.O_RDONLY | os.O_DIRECT); bufs = [mmap.mmap(-1, PG) for _ in range(T)]
    def rd(i): os.preadv(fd, [bufs[i % T]], o[i])
    t = time.perf_counter()
    with ThreadPoolExecutor(T) as ex: list(ex.map(rd, range(len(o))))
    dt = time.perf_counter() - t; os.close(fd)
    print(f"O_DIRECT pread {T:2d} threads: {len(o)/dt:9.0f} IOPS", flush=True)
# 2) the prefetcher's pattern: numpy row views on a MADV_RANDOM map, rows grouped into tasks, 64 threads
fd = os.open(F, os.O_RDONLY); m = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); os.close(fd); m.madvise(mmap.MADV_RANDOM)
arr = np.frombuffer(m, dtype=np.uint8); RB = 160; rows = (SZ - 8192) // RB
view = arr[8192: 8192 + rows * RB].reshape(rows, RB)
for T in (1, 8, 64):
    drop(); r = np.random.default_rng(T).integers(0, rows, 8000); tasks = np.array_split(np.sort(r), 64)
    def fault(ix): return int(view[ix, 0].sum()) + int(view[ix, RB - 1].sum())
    t = time.perf_counter()
    with ThreadPoolExecutor(T) as ex: list(ex.map(fault, tasks))
    dt = time.perf_counter() - t
    print(f"numpy row-touch (prefetcher pattern) {T:2d} threads: {len(r)/dt:9.0f} rows/s", flush=True)
