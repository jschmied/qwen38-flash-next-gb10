"""Exit 0 only if a held-out recording is complete: one finished run per held-out conversation, the expected rows,
and assistant rows marked (kind 1)."""

import json
import sys
from pathlib import Path

import numpy as np

rec_dir, prefix, names_file, cap = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
names = json.loads(Path(names_file).read_text())
idx = json.loads(Path(prefix + ".idx.json").read_text())
want = {m["source"]: min(m["tokens"], cap) for m in idx["meta"] if m["source"] in names}
got = {}
for meta in rec_dir.rglob("*.json"):
    info = json.loads(meta.read_text())
    kind = np.fromfile(str(meta)[:-5] + ".kind", dtype=np.uint8)
    got[info.get("source")] = (info["rows"], len(kind), int(kind.sum()))
bad = [n for n in names if n not in got or got[n][0] != want[n] or got[n][1] != want[n] or got[n][2] == 0]
print(json.dumps({"want": want, "got": got, "bad": bad}))
sys.exit(1 if bad or len(want) != len(names) else 0)
