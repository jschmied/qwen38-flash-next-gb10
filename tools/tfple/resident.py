#!/usr/bin/env python3
"""Page-cache residency of a file (mincore over a read-only mapping; reads nothing). argv: <file> [interval_s count]
Prints one line per sample: time, resident GiB, percent."""
import ctypes, mmap, sys, time

import numpy as np

path = sys.argv[1]
every, count = (float(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
libc = ctypes.CDLL(None, use_errno=True)
libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p)
view = np.memmap(path, dtype=np.uint8, mode="r")
start, size = view.__array_interface__["data"][0], view.nbytes
pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
vec = np.zeros(pages, dtype=np.uint8)
for i in range(count):
    if libc.mincore(start, size, vec.ctypes.data) != 0:
        sys.exit(f"mincore failed: errno {ctypes.get_errno()}")
    res = int(np.count_nonzero(vec & 1))
    print(f"{time.strftime('%T')} resident {res * mmap.PAGESIZE / 2**30:.2f} GiB {100 * res / pages:.1f} %", flush=True)
    if i < count - 1:
        time.sleep(every)
