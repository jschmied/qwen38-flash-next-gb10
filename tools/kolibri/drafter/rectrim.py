"""Trim finished recordings to what training reads: the `keep` rows before the first row Kolibri generated, and
everything after it. Training anchors only on generated rows and reads back one window (2,048 rows), so the head of a
long agent prompt (recorded again on every turn when the prompt cache misses) is never used.

    python rectrim.py DIR [--keep 2048] [--loop 600]

Only recordings with a .json (finished) are touched; each file is rewritten to a .tmp and renamed, the .json last, with
`trimmed_from` (rows dropped) added, so a reader sees either the old recording or the new one, and a trimmed one is
skipped next time.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PARTS = {".tok": (np.int32, 1), ".pos": (np.int32, 1), ".kind": (np.uint8, 1)}


def trim(meta: Path, keep: int) -> int:
    info = json.loads(meta.read_text())
    if "trimmed_from" in info:
        return 0
    base, n = str(meta)[:-5], info["rows"]
    kind = np.fromfile(base + ".kind", dtype=np.uint8, count=n)
    gen = np.flatnonzero(kind)
    start = max(0, int(gen[0]) - keep) if len(gen) else max(0, n - keep)
    if start == 0:
        info["trimmed_from"] = 0
    else:
        width = len(info["taps"]) * info["dims"]
        shapes = {".st": (np.int16, width), ".tki": (np.int32, info["k"]), ".tkl": (np.int16, info["k"]), **PARTS}
        for suf, (dt, w) in shapes.items():
            if not os.path.exists(base + suf):
                continue
            a = np.memmap(base + suf, dtype=dt, mode="r", shape=(n, w) if w > 1 else (n,))
            np.ascontiguousarray(a[start:]).tofile(base + suf + ".tmp")
            del a
            os.replace(base + suf + ".tmp", base + suf)
        info["rows"] = n - start
        info["trimmed_from"] = start
    tmp = meta.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(info))
    os.replace(tmp, meta)
    return start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--keep", type=int, default=2048)
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes (0: one pass)")
    a = ap.parse_args()
    while True:
        dropped, done = 0, 0
        for meta in sorted(Path(a.dir).rglob("*.json")):
            if meta.name.endswith(".tmp"):
                continue
            rows = trim(meta, a.keep)
            dropped += rows
            done += rows > 0
        print(json.dumps({"time": time.strftime("%H:%M"), "trimmed": done, "rows_dropped": dropped}), flush=True)
        if not a.loop:
            break
        time.sleep(a.loop)


if __name__ == "__main__":
    main()
