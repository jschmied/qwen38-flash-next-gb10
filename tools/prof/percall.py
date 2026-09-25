"""Per-call attribution for one kernel class: duration vs gap-before, predecessor, position. argv: dir cpuops.tsv match grid"""
import sys, bisect, collections, statistics as st
D, CPU, MATCH, GRID = sys.argv[1:5]
A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]
K = []
for l in open(f"{D}/kernels2.tsv"):
    c, ts, d, s, g, b, n = l.rstrip("\n").split("\t", 6)
    t = float(ts)
    if lo <= t < hi: K.append((t, float(d), s, g, n))
K.sort()
ends = []  # running max end of ANY kernel before index i
m = 0
for t, d, *_ in K:
    ends.append(m); m = max(m, t + d)
rows = []
main = [i for i, k in enumerate(K) if k[2] == "33"]
pos_in_step = {}
for j, i in enumerate(main):
    t, d, s, g, n = K[i]
    if MATCH in n and g == GRID:
        gap = t - ends[i]
        pj = main[j - 1] if j else None
        pred = K[pj][4][:60] if pj is not None else "-"
        pgrid = K[pj][3] if pj is not None else "-"
        step = bisect.bisect_right(gen, t) - 1
        pos_in_step.setdefault(step, 0); k_in = pos_in_step[step]; pos_in_step[step] += 1
        rows.append((d, gap, pred + " " + pgrid, k_in))
ds = sorted(r[0] for r in rows)
q = lambda p: round(ds[int(p * (len(ds) - 1))], 1)
print(f"n={len(rows)} dur p5 {q(.05)} p25 {q(.25)} p50 {q(.5)} p75 {q(.75)} p95 {q(.95)} mean {round(st.mean(ds),1)}")
hist = collections.Counter(int(x // 2) * 2 for x in ds)
print("hist (2us bins):", " ".join(f"{k}:{v}" for k, v in sorted(hist.items()) if v > len(ds) * 0.01))
for lab, sel in (("gap<=1us", lambda r: r[1] <= 1), ("gap 1-5us", lambda r: 1 < r[1] <= 5), ("gap>5us", lambda r: r[1] > 5)):
    v = [r[0] for r in rows if sel(r)]
    if v: print(f"{lab:10s} n={len(v):5d} median dur {round(st.median(v),1)}")
pc = collections.defaultdict(list)
for r in rows: pc[r[2]].append(r[0])
print("by predecessor (n>=100):")
for p, v in sorted(pc.items(), key=lambda x: -len(x[1])):
    if len(v) >= 100: print(f"  n={len(v):5d} median {round(st.median(v),1):5}  {p}")
bypos = collections.defaultdict(list)
for r in rows: bypos[r[3]].append(r[0])
print("by call index within step (every 12th):", [(k, round(st.median(v), 1)) for k, v in sorted(bypos.items()) if k % 12 == 0])
