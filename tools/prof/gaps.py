"""Where does GPU idle sit, and which small-kernel chains sit between big kernels (warm trace)."""
import sys, bisect, collections, re
D, CPU = sys.argv[1:3]
A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]; n = len(gen) - 1 - 5
ev = []
for l in open(f"{D}/kernels2.tsv"):
    c, ts, d, s, g, b, nm = l.rstrip("\n").split("\t", 6); t = float(ts)
    if lo <= t < hi: ev.append((t, t + float(d), s, g, re.sub(r"<.*", "", nm)[:38]))
ev.sort()
# 1) idle gaps in the union of all streams, attributed to (last kernel before, first kernel after)
gaps = collections.Counter(); gapn = collections.Counter(); end = ev[0][1]; last = ev[0][4]; total = 0
for t0, t1, s, g, nm in ev[1:]:
    if t0 > end + 1.0:
        k = (last, nm); gaps[k] += t0 - end; gapn[k] += 1; total += t0 - end
    if t1 > end: end, last = t1, nm
print(f"idle (union gaps > 1 us): {total/n/1e3:.2f} ms/step; top pairs:")
for k, v in gaps.most_common(12): print(f"  {v/n/1e3:6.3f} ms/step  n/step={gapn[k]/n:5.1f}  {k[0]}  ->  {k[1]}")
# 2) small-kernel chains on the main stream between kernels >= 20 us
main = [e for e in ev if e[2] == "33"]
chains = collections.Counter(); chain_t = collections.Counter(); cur = []
def flush():
    if len(cur) >= 2:
        key = " > ".join(x[4] for x in cur[:6]) + (" ..." if len(cur) > 6 else "")
        chains[key] += 1; chain_t[key] += (cur[-1][1] - cur[0][0])
for e in main:
    if e[1] - e[0] < 20: cur.append(e)
    else: flush(); cur = []
flush()
tot = sum(chain_t.values())
print(f"\nsmall-kernel chains (>=2 consecutive <20us kernels, main stream): {tot/n/1e3:.2f} ms/step wall incl. gaps; top:")
for k, v in sorted(chain_t.items(), key=lambda x: -x[1])[:12]:
    print(f"  {v/n/1e3:6.3f} ms/step  n/step={chains[k]/n:5.1f}  avg {v/chains[k]:6.1f} us  {k}")
