"""FNFILL2: FN_PLE_FILL=nowait replays the #58439 follow-up PR exactly on the older prod-based stack: decode-sized
row sets (< 4096 rows) are filled with MADV_WILLNEED + MADV_POPULATE_READ per page instead of the serial touch, with
no sync-touch wait. Applied on top of patch_fill.py (FNFILL). argv[1] = ple_pageable.py; argv[2] == "off" removes."""
import sys
p = sys.argv[1]; s = open(p).read(); MARK = "FNFILL2"
EDITS = [
    ("""        if pool is None or rows.size <= 4096:
            for task in groups(np.arange(rows.size)):
                fault(task)
            return""",
     """        if _FILL_NOWAIT and rows.size <= 4096:  # FNFILL2: the follow-up PR's path, no wait
            pages = []
            for view, index in groups(np.arange(rows.size)):
                start = view.__array_interface__["data"][0] + index.astype(np.int64) * self.row_bytes
                pages.append(start & ~4095)
                pages.append((start + (self.row_bytes - 1)) & ~4095)
            for p_ in (np.unique(np.concatenate(pages)).tolist() if pages else []):
                _POP_LIBC.madvise(ctypes.c_void_p(p_), 4096, 3)            # MADV_WILLNEED
            for p_ in (np.unique(np.concatenate(pages)).tolist() if pages else []):
                _POP_LIBC.madvise(ctypes.c_void_p(p_), 4096, _MADV_POPULATE_READ)
            return
        if pool is None or rows.size <= 4096:
            for task in groups(np.arange(rows.size)):
                fault(task)
            return"""),
    ("""_FILL_WILLNEED = os.environ.get("FN_PLE_FILL", "") == "willneed"   # FNFILL""",
     """_FILL_WILLNEED = os.environ.get("FN_PLE_FILL", "") == "willneed"   # FNFILL
_FILL_NOWAIT = os.environ.get("FN_PLE_FILL", "") == "nowait"   # FNFILL2
if _FILL_NOWAIT:
    logger.warning("FNFILL2 armed: decode pages filled by readahead, no wait (the #58439 follow-up)")"""),
]
if len(sys.argv) > 2 and sys.argv[2] == "off":
    for old, new in EDITS:
        s = s.replace(new, old)
    assert MARK not in s; open(p, "w").write(s); print("removed"); sys.exit()
if MARK in s:
    print("already"); sys.exit()
for old, new in EDITS:
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)
open(p, "w").write(s); print("patched")
