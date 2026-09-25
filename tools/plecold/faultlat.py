import os, mmap, random, time, statistics as st, ctypes
from concurrent.futures import ThreadPoolExecutor
F = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-00005.safetensors"
SZ = os.path.getsize(F); PG = 4096; N = 3000
libc = ctypes.CDLL("libc.so.6"); libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def drop():
    fd = os.open(F, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
def offs(n, seed): r = random.Random(seed); return [r.randrange(SZ // PG) * PG for _ in range(n)]
def summ(name, lat):
    lat = sorted(x * 1e6 for x in lat)
    print(f"{name:34s} p50 {lat[len(lat)//2]:7.1f} us  p90 {lat[int(.9*len(lat))]:7.1f}  p99 {lat[int(.99*len(lat))]:7.1f}  mean {st.mean(lat):7.1f}", flush=True)
# A: O_DIRECT
drop(); buf = mmap.mmap(-1, PG); fd = os.open(F, os.O_RDONLY | os.O_DIRECT); lat = []
for o in offs(N, 1):
    t = time.perf_counter(); os.preadv(fd, [buf], o); lat.append(time.perf_counter() - t)
os.close(fd); summ("A O_DIRECT pread 4K (SSD)", lat)
# B: buffered pread, readahead off
drop(); fd = os.open(F, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM); lat = []
for o in offs(N, 2):
    t = time.perf_counter(); os.pread(fd, PG, o); lat.append(time.perf_counter() - t)
os.close(fd); summ("B buffered pread 4K (page cache)", lat)
# C: mmap MADV_RANDOM major fault
drop(); fd = os.open(F, os.O_RDONLY); m = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); os.close(fd)
m.madvise(mmap.MADV_RANDOM); lat = []
for o in offs(N, 3):
    t = time.perf_counter(); m[o]; lat.append(time.perf_counter() - t)
summ("C mmap major fault (MADV_RANDOM)", lat)
# E: minor fault: pages cached (from C), fresh mapping
fd = os.open(F, os.O_RDONLY); m2 = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); os.close(fd); m2.madvise(mmap.MADV_RANDOM); lat = []
for o in offs(N, 3):
    t = time.perf_counter(); m2[o]; lat.append(time.perf_counter() - t)
summ("E mmap minor fault (page cached)", lat)
# D: 64 threads of major faults, like the PagePrefetcher
drop(); fd = os.open(F, os.O_RDONLY); m3 = mmap.mmap(fd, SZ, mmap.MAP_SHARED, mmap.PROT_READ); os.close(fd); m3.madvise(mmap.MADV_RANDOM)
for T in (1, 8, 64):
    o = offs(4000, 10 + T); drop()
    t = time.perf_counter()
    with ThreadPoolExecutor(T) as ex: list(ex.map(lambda x: m3[x], o))
    dt = time.perf_counter() - t
    print(f"D mmap faults, {T:2d} threads: {len(o)/dt:8.0f} faults/s = {1e6*dt/len(o):6.1f} us/fault effective", flush=True)
