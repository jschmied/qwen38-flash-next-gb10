import os, ctypes, glob, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE"); tot = res = 0
for f in sorted(glob.glob(sys.argv[1])):
    sz = os.path.getsize(f); fd = os.open(f, os.O_RDONLY)
    a = libc.mmap(None, sz, 1, 1, fd, 0)                       # PROT_READ, MAP_SHARED
    n = (sz + PAGE - 1) // PAGE; vec = (ctypes.c_ubyte * n)()
    rc = libc.mincore(a, sz, vec)
    r = sum(1 for v in vec if v & 1); tot += sz; res += r * PAGE
    print(os.path.basename(f), f"{sz/2**30:.2f} GiB", f"resident {r*PAGE/2**30:.3f} GiB ({100*r*PAGE/sz:.1f} %)", "rc", rc)
    libc.munmap(a, sz); os.close(fd)
print(f"TOTAL resident {res/2**30:.2f} of {tot/2**30:.2f} GiB")
