#!/usr/bin/env python3
"""Where does time go at the QSA kernel border? From a torch-profiler trace, find each QSA
attention call (the split-K kernel or the tile-union attention kernel) and look at the window
from the end of the indexer's top-k kernel to the attention kernel's end: which kernels ran in
between (expansion, layout ops, pack, sort, build), how much GPU time they took, and how much of
the window was idle (host-side glue, launches, syncs). Usage: prof_border.py <trace> [--all]"""
import gzip, json, sys
from collections import defaultdict

path = sys.argv[1]
show_all = "--all" in sys.argv
opener = gzip.open if path.endswith(".gz") else open
with opener(path, "rt") as f:
    ev = json.load(f)["traceEvents"]
kernels = sorted((e for e in ev if e.get("cat") == "kernel" and "dur" in e), key=lambda e: e["ts"])
ATTN = ("_qsa_sparse_paged_gqa_splitk_kernel", "_qsa_tile_union_attn_kernel")
TOPK = ("_qsa_mqa_paged_prefill_kernel", "_qsa_mqa_paged_decode_kernel", "persistent_topk", "topk")


def short(n):
    return n if len(n) < 60 else n[:57] + "..."


windows = []
for i, k in enumerate(kernels):
    if not any(k["name"].startswith(a) for a in ATTN):
        continue
    # walk back to the top-k kernel that fed this call
    j = i - 1
    while j >= 0 and not any(t in kernels[j]["name"] for t in TOPK):
        j -= 1
    if j < 0:
        continue
    start = kernels[j]["ts"] + kernels[j]["dur"]
    end = k["ts"] + k["dur"]
    inside = [e for e in kernels[j + 1 : i + 1]]
    busy = sum(e["dur"] for e in inside)
    windows.append((start, end, k, inside, busy))

print(f"{len(windows)} attention calls; window = top-k kernel end -> attention kernel end")
agg = defaultdict(float)
tot_win = tot_busy = tot_attn = 0.0
for start, end, k, inside, busy in windows:
    tot_win += end - start
    tot_busy += busy
    tot_attn += k["dur"]
    for e in inside:
        if e is not k:
            agg[short(e["name"])] += e["dur"]
idle = tot_win - tot_busy
print(f"windows total {tot_win / 1e3:8.1f} ms | kernels inside {tot_busy / 1e3:8.1f} ms | attention kernel {tot_attn / 1e3:8.1f} ms | "
      f"glue kernels {(tot_busy - tot_attn) / 1e3:8.1f} ms | GPU idle in window {idle / 1e3:8.1f} ms ({100 * idle / tot_win:.1f} %)")
print("kernels between top-k and attention (summed over calls):")
for name, dur in sorted(agg.items(), key=lambda kv: -kv[1])[:20]:
    print(f"    {name:60s} {dur / 1e3:8.2f} ms")
if show_all:
    for n, (start, end, k, inside, busy) in enumerate(windows):
        print(f"  call {n:2d}: window {(end - start) / 1e3:7.2f} ms, busy {busy / 1e3:7.2f}, attn {k['dur'] / 1e3:7.2f}, idle {(end - start - busy) / 1e3:6.2f}")
# gaps immediately before the attention kernel (host launch latency for the big kernel)
gaps = []
for start, end, k, inside, busy in windows:
    prev = [e for e in inside if e is not k]
    if prev:
        p = max(prev, key=lambda e: e["ts"] + e["dur"])
        gaps.append(k["ts"] - (p["ts"] + p["dur"]))
if gaps:
    gaps.sort()
    print(f"gap between the last glue kernel and the attention kernel: median {gaps[len(gaps) // 2]:.0f} us, max {gaps[-1]:.0f} us")
