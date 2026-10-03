#!/usr/bin/env python3
"""GPU idle in an nsys capture: total idle (no kernel or memcpy on any stream), idle by the kernel that follows the gap,
the largest gaps with the kernels around them and the CUDA runtime calls that overlap each gap. argv: <capture.sqlite>"""
import collections, re, sqlite3, sys

c = sqlite3.connect(sys.argv[1])
S = dict(c.execute("select id, value from StringIds"))
ev = [(s, e, re.sub(r"<.*|\(.*", "", S.get(n, "?"))[:60], "k") for s, e, n in
      c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
try:
    ev += [(s, e, f"memcpy kind {k}", "m") for s, e, k in c.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY")]
except sqlite3.OperationalError:
    pass
ev.sort()
rt = sorted((s, e, S.get(n, "?")) for s, e, n in c.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"))
gaps, cur_end, prev = [], ev[0][1], ev[0]
for x in ev[1:]:
    if x[0] > cur_end:
        gaps.append((x[0] - cur_end, cur_end, x[0], prev[2], x[2]))
    if x[1] > cur_end:
        cur_end, prev = x[1], x
span = ev[-1][1] - ev[0][0]
idle = sum(g[0] for g in gaps)
print(f"span {span / 1e6:.1f} ms, GPU idle {idle / 1e6:.1f} ms in {len(gaps)} gaps")
by_next = collections.Counter()
for g in gaps:
    by_next[g[4]] += g[0]
print("idle before the next kernel (top 12):")
for k, v in by_next.most_common(12):
    print(f"  {v / 1e6:8.1f} ms  {k}")
hist = collections.Counter()
for g in gaps:
    hist["<10us" if g[0] < 1e4 else "10-100us" if g[0] < 1e5 else "0.1-1ms" if g[0] < 1e6 else ">1ms"] += g[0]
print("idle by gap size:", {k: round(v / 1e6, 1) for k, v in hist.items()})
print("largest gaps:")
import bisect
starts = [r[0] for r in rt]
for d, a, b, before, after in sorted(gaps, reverse=True)[:15]:
    i = bisect.bisect_left(starts, a - 50_000_000)
    calls = collections.Counter()
    for s, e, n in rt[i:]:
        if s > b:
            break
        if e > a:
            calls[n.split("_v")[0]] += (min(e, b) - max(s, a))
    top = ", ".join(f"{n} {t / 1e3:.0f}us" for n, t in calls.most_common(3))
    print(f"  {d / 1e3:8.0f} us  after {before} -> before {after} | host: {top}")
