#!/usr/bin/env python3
"""Summarise a vLLM torch-profiler trace (*.pt.trace.json[.gz]): GPU kernel time by kernel
name, total span, and the share of the QSA / tile-union kernels. Usage: prof_summary.py <trace> [top]"""
import gzip, json, sys
from collections import defaultdict

path = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    events = json.load(f)["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel" and "dur" in e]
by_name = defaultdict(lambda: [0.0, 0])
for e in kernels:
    by_name[e["name"]][0] += e["dur"]
    by_name[e["name"]][1] += 1
total = sum(v[0] for v in by_name.values())
t0 = min(e["ts"] for e in kernels)
t1 = max(e["ts"] + e["dur"] for e in kernels)
print(f"kernel-sum {total / 1e6:.3f} s over span {(t1 - t0) / 1e6:.3f} s, {len(kernels)} launches")
rows = sorted(by_name.items(), key=lambda kv: -kv[1][0])
print(f"{'kernel':78s} {'ms':>9s} {'%':>6s} {'calls':>6s} {'us/call':>9s}")
for name, (dur, calls) in rows[:top]:
    print(f"{name[:78]:78s} {dur / 1e3:9.1f} {100 * dur / total:6.1f} {calls:6d} {dur / calls:9.1f}")
qsa = sum(v[0] for k, v in by_name.items() if "qsa" in k.lower() or "tile_union" in k.lower())
print(f"QSA-related kernels: {qsa / 1e3:.1f} ms = {100 * qsa / total:.1f} % of kernel time")
for k, v in rows:
    if "qsa" in k.lower() or "tile_union" in k.lower():
        print(f"    {k[:78]:78s} {v[0] / 1e3:9.1f} ms {v[1]:6d} calls")
