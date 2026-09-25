#!/usr/bin/env python3
"""Warm the checkpoint-mapped PLE page cache (sequential read of the 10 shards), then exec the given probe.
Warm stats go to stderr so the probe's single stdout JSON (armrun contract) stays intact. argv: probe.py args..."""
import glob, os, sys, time
PLE = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-*.safetensors"
t0 = time.time(); buf = bytearray(64 << 20); mv = memoryview(buf); n = 0
for f in sorted(glob.glob(PLE)):
    fd = os.open(f, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
        while (r := os.readv(fd, [mv])) > 0: n += r
    finally:
        os.close(fd)
m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
print(f"WARM read {n/2**30:.1f} GiB in {time.time()-t0:.1f} s; MemAvailable {m['MemAvailable']/2**20:.1f} GiB, Cached {m['Cached']/2**20:.1f}", file=sys.stderr, flush=True)
os.execv(sys.executable, [sys.executable] + sys.argv[1:])
