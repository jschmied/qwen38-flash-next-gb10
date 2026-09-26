"""FNFILL: FN_PLE_FILL=willneed fills the step's pages with MADV_WILLNEED (async readahead for every page) followed by
MADV_POPULATE_READ per page, both via ctypes from the prefetch thread (GIL released in each call). No compiled
helper. argv[1] = path of ple_pageable.py; argv[2] == "off" removes it."""
import sys
p = sys.argv[1]; s = open(p).read(); MARK = "FNFILL"
A_OLD = """                if _POP_C is not None:
                    _POP_C.fn_populate(pages.ctypes.data, int(pages.size), 16)"""
A_NEW = """                if _FILL_WILLNEED:  # FNFILL: readahead all, then wait page by page (GIL released in each call)
                    for p_ in pages:
                        _POP_LIBC.madvise(ctypes.c_void_p(int(p_)), 4096, 3)       # MADV_WILLNEED
                    for p_ in pages:
                        _POP_LIBC.madvise(ctypes.c_void_p(int(p_)), 4096, _MADV_POPULATE_READ)
                elif _POP_C is not None:
                    _POP_C.fn_populate(pages.ctypes.data, int(pages.size), 16)"""
B_OLD = """_POP_C = None
if _SYNCTOUCH:"""
B_NEW = """_POP_C = None
_FILL_WILLNEED = os.environ.get("FN_PLE_FILL", "") == "willneed"   # FNFILL
if _SYNCTOUCH:"""
C_OLD = """_POP_C is not None, _SYNCMODE, _SYNC_THRESHOLD)"""
C_NEW = """_POP_C is not None, _SYNCMODE, _SYNC_THRESHOLD)
            if _FILL_WILLNEED:
                logger.warning("FNFILL armed: page fill = MADV_WILLNEED readahead + MADV_POPULATE_READ, no C helper")"""
if len(sys.argv) > 2 and sys.argv[2] == "off":
    for old, new in ((A_OLD, A_NEW), (B_OLD, B_NEW), (C_OLD, C_NEW)):
        s = s.replace(new, old)
    assert MARK not in s; open(p, "w").write(s); print("removed"); sys.exit()
if MARK in s:
    print("already"); sys.exit()
for old, new in ((A_OLD, A_NEW), (B_OLD, B_NEW), (C_OLD, C_NEW)):
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
open(p, "w").write(s); print("patched")
