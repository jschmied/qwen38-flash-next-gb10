import sys, bisect, collections, statistics as st
D, CPU = sys.argv[1:3]
A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]
ev = []
for l in open(f"{D}/kernels2.tsv"):
    c, ts, d, s, g, b, n = l.rstrip("\n").split("\t", 6); t = float(ts)
    if lo - 2e5 <= t < hi: ev.append((t, t + float(d), c, s, g, n))
ev.sort()
starts = [e[0] for e in ev]
main = [e for e in ev if e[3] == "33"]
def concurrent(a, b):
    out = collections.Counter()
    i = bisect.bisect_left(starts, a - 5000)
    while i < len(ev) and ev[i][0] < b:
        t0, t1, c, s, g, n = ev[i]
        if s != "33" and t1 > a:
            out[f"{c}:{n[:40]}"] += min(b, t1) - max(a, t0)
        i += 1
    return out
groups = {"slow(after out_proj)": [], "fast": []}
for j, (t0, t1, c, s, g, n) in enumerate(main):
    if not (lo <= t0 < hi and (("wmma" in n and g == "[8,3,9]") or ("_partial" in n and g == "[21,2,1]"))): continue
    prev = [main[j - q][5] for q in range(1, 4) if j - q >= 0]
    slow = any("fp8_blockwise" in p for p in prev)
    # window: from start of the preceding out_proj (up to 3 kernels back) to end of this kernel
    w0 = main[j - 2][0] if slow else t0
    groups["slow(after out_proj)" if slow else "fast"].append((t1 - t0, concurrent(t0, t1), concurrent(w0, t1)))
for k, v in groups.items():
    tot = collections.Counter(); totw = collections.Counter()
    for d, c, cw in v: tot.update(c); totw.update(cw)
    print(f"== {k}: n={len(v)} median {st.median(x[0] for x in v):.1f} us; concurrent other-stream time per call:")
    for n, x in tot.most_common(5): print(f"     during kernel  {x/len(v):7.2f} us  {n}")
    for n, x in totw.most_common(5): print(f"     incl. out_proj {x/len(v):7.2f} us  {n}")
