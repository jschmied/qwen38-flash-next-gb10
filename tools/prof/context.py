import sys, bisect, collections, statistics as st
D, CPU, MATCH, GRID = sys.argv[1:5]
A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]
K = []
for l in open(f"{D}/kernels2.tsv"):
    c, ts, d, s, g, b, n = l.rstrip("\n").split("\t", 6); t = float(ts)
    if lo - 1e5 <= t < hi: K.append((t, float(d), s, g, n))
K.sort()
side = sorted((t, t + d, n) for t, d, s, g, n in K if s != "33")
side_end = sorted(e for _, e, _ in side)
main = [k for k in K if k[2] == "33"]
cnt = collections.Counter(); grp = {0: [], 1: []}
for j, (t, d, s, g, n) in enumerate(main):
    if not (lo <= t < hi and MATCH in n and g == GRID): continue
    stp = bisect.bisect_right(gen, t) - 1; idx = cnt[stp]; cnt[stp] += 1
    last_side = side_end[bisect.bisect_right(side_end, t) - 1] if side_end and side_end[0] < t else None
    since_side = t - last_side if last_side else 1e9
    prev = [f"{main[j-q][4][:28]}{main[j-q][3]}({main[j-q][1]:.0f})" for q in range(4, 0, -1)]
    grp[idx % 2].append((d, since_side, prev))
for par in (0, 1):
    v = grp[par]
    print(f"== {'EVEN' if par == 0 else 'ODD'} n={len(v)} median dur {st.median(x[0] for x in v):.1f}  median us since last side-stream kernel end {st.median(x[1] for x in v):.1f}")
    c = collections.Counter(" | ".join(x[2]) for x in v)
    for p, n in c.most_common(2): print(f"   n={n}: {p}")
