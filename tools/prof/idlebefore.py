import sys, bisect, collections, statistics as st
D, CPU = sys.argv[1:3]
A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]
ev = []
for l in open(f"{D}/kernels2.tsv"):
    c, ts, d, s, g, b, n = l.rstrip("\n").split("\t", 6); t = float(ts)
    if lo - 2e5 <= t < hi: ev.append((t, t + float(d), s, g, n))
ev.sort()
# merged busy intervals over all streams
busy = []
for a, b, *_ in ev:
    if busy and a <= busy[-1][1]: busy[-1][1] = max(busy[-1][1], b)
    else: busy.append([a, b])
bs = [x[0] for x in busy]
def idle(a, b):
    i = max(bisect.bisect_right(bs, a) - 1, 0); cov = 0
    while i < len(busy) and busy[i][0] < b:
        cov += max(0, min(b, busy[i][1]) - max(a, busy[i][0])); i += 1
    return (b - a) - cov
main = [e for e in ev if e[2] == "33"]
rows = collections.defaultdict(list)
for j, (t0, t1, s, g, n) in enumerate(main):
    if not lo <= t0 < hi: continue
    if "fp8_blockwise" in n and g == "[1,20,1]":
        kind = "out_proj(GDN)" if any("fused_sigmoid_gating" in main[j-q][4] for q in range(1, 12)) else "o_proj(QSA)"
    elif ("wmma" in n and g == "[8,3,9]") or ("_partial" in n and g == "[21,2,1]"):
        kind = "mixer_down(slow site)" if any("fp8_blockwise" in main[j-q][4] for q in range(1, 4)) else "mixer_down(fast site)"
    elif "fp8_blockwise" in n and g == "[1,128,1]":
        kind = "in_proj_qkvz"
    else: continue
    rows[kind].append((t1 - t0, idle(t0 - 200, t0), idle(t0 - 1000, t0)))
for k, v in rows.items():
    v.sort(key=lambda r: r[2]); q = len(v) // 4
    lo_q, hi_q = v[:q], v[-q:]
    print(f"{k:24s} n={len(v):5d} dur med {st.median(r[0] for r in v):6.1f} | idle in prior 200us med {st.median(r[1] for r in v):5.1f}, prior 1ms med {st.median(r[2] for r in v):6.1f} "
          f"| dur when prior-1ms idle lowest quartile {st.median(r[0] for r in lo_q):6.1f} vs highest {st.median(r[0] for r in hi_q):6.1f}")
