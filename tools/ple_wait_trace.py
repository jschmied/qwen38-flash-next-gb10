#!/usr/bin/env python3
"""Per-decode-step PLE offload wait from a vLLM torch-profiler trace.

Steps are delimited by the connector's CPU marker "ple_offload: enqueue_cuda_inputs" (one per target forward). For each
step: GPU busy time, GPU idle time, the time from the step's first kernel to the first PLE kernel (n-gram ids / conv), and
the idle gap immediately before that PLE kernel — the part of the host gather + H2D that the GPU could not hide behind
the embedding and layers 0–1. Usage: ple_wait_trace.py <trace.json[.gz]> [min_step_kernels]"""
import gzip, json, sys, statistics
path = sys.argv[1]; min_k = int(sys.argv[2]) if len(sys.argv) > 2 else 50
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f: ev = json.load(f)["traceEvents"]
marks = sorted(e["ts"] for e in ev if e.get("cat") in ("cpu_op", "user_annotation") and "ple_offload: enqueue_cuda_inputs" in e.get("name", ""))
kern = sorted((e for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda e: e["ts"])
print(f"markers={len(marks)} kernels={len(kern)}")
rows = []
for i, m in enumerate(marks):
    end = marks[i + 1] if i + 1 < len(marks) else float("inf")
    ks = [k for k in kern if m <= k["ts"] < end]
    if len(ks) < min_k: continue
    t0 = ks[0]["ts"]; t1 = max(k["ts"] + k["dur"] for k in ks); busy = 0.0; cur = t0; gaps = []
    for k in ks:
        if k["ts"] > cur: gaps.append((k["ts"] - cur, k["name"], k["ts"]))
        busy += min(k["dur"], max(0.0, k["ts"] + k["dur"] - max(k["ts"], cur))); cur = max(cur, k["ts"] + k["dur"])
    ple = [k for k in ks if "ple" in k["name"].lower() or "ngram" in k["name"].lower()]
    if not ple: continue
    p0 = ple[0]; gap_before = next((g for g, n, ts in gaps if ts == p0["ts"]), 0.0)
    rows.append(dict(span=t1 - t0, busy=busy, idle=(t1 - t0) - busy, to_ple=p0["ts"] - t0, gap_before_ple=gap_before, nk=len(ks), first_ple=p0["name"][:50]))
if not rows: print("no decode steps with PLE kernels found"); sys.exit(0)
def med(k): return statistics.median(r[k] for r in rows)
print(f"steps={len(rows)}  first PLE kernel: {rows[0]['first_ple']}")
print(f"median per step: span {med('span'):.0f} us | GPU busy {med('busy'):.0f} us | idle {med('idle'):.0f} us | time to first PLE kernel {med('to_ple'):.0f} us | idle gap right before it {med('gap_before_ple'):.0f} us")
print(f"max gap before PLE {max(r['gap_before_ple'] for r in rows):.0f} us; steps with gap > 200 us: {sum(r['gap_before_ple'] > 200 for r in rows)}")
