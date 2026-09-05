#!/usr/bin/env python3
"""Attribute GPU kernels in a vLLM torch-profiler trace to the CPU op / module that launched them.

For every kernel whose name matches a pattern, follow the launch correlation to its CPU op, then find the innermost
enclosing user_annotation / python_function on that thread (module forward names), and sum kernel time per
(kernel family, launching op, enclosing annotation). Usage: prof_attrib.py <trace.json[.gz]> <regex> [top]"""
import gzip, json, re, sys
from collections import defaultdict
path, pat = sys.argv[1], re.compile(sys.argv[2]); top = int(sys.argv[3]) if len(sys.argv) > 3 else 20
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f: ev = json.load(f)["traceEvents"]
kern = [e for e in ev if e.get("cat") == "kernel" and pat.search(e.get("name", ""))]
ext_cpu = {}
for e in ev:
    if e.get("cat") == "cpu_op" and "args" in e and "External id" in e["args"]:
        ext_cpu.setdefault(e["args"]["External id"], e)
pyf = defaultdict(list)  # tid -> [(ts, te, name)] python_function frames that name a vLLM module forward
for e in ev:
    if e.get("cat") == "python_function" and "dur" in e and "vllm" in e.get("name", "") and "torch/" not in e["name"]:
        pyf[e.get("tid")].append((e["ts"], e["ts"] + e["dur"], e["name"]))
for t in pyf: pyf[t].sort()
def enclosing(tid, ts):
    best = None
    for s, te, name in pyf.get(tid, []):
        if s > ts: break
        if s <= ts <= te and (best is None or (te - s) < (best[1] - best[0])): best = (s, te, name)
    return best[2].split("/")[-1] if best else "?"
agg = defaultdict(lambda: [0.0, 0])
for k in kern:
    op = ext_cpu.get(k.get("args", {}).get("External id"))
    launch_op = op["name"] if op else "?"; tid = op.get("tid") if op else None; ts = op["ts"] if op else k["ts"]
    dims = str(op.get("args", {}).get("Input Dims", ""))[:44] if op else ""
    key = (k["name"][:34], launch_op[:12] + " " + dims, enclosing(tid, ts)[:52])
    agg[key][0] += k["dur"]; agg[key][1] += 1
tot = sum(v[0] for v in agg.values()); print(f"matched kernels: {len(kern)}, {tot/1e3:.1f} ms")
for key, (d, n) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{d/1e3:8.1f} ms {n:5d}  {key[0]:40s} | {key[1]:30s} | {key[2]}")
