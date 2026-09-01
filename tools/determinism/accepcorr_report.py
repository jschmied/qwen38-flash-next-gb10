import re, sys
# usage: accepcorr_report.py <results.txt>   (the arm log produced by the accepcorr runner)
t = open(sys.argv[1] if len(sys.argv) > 1 else "accepcorr.txt", errors="replace").read()
ms=dict(re.findall(r"^ *(AC\d_n5) TOTAL .*?([\d.]+) ms/tok",t,re.M))
al=dict((m[0],m[1]) for m in re.findall(r"ACCEPT (AC\d_n5)-post:? .*?mean_accept_len=([\d.]+)",t))
rows=[(k,float(ms[k]),float(al[k])) for k in sorted(ms) if k in al]
for k,a,b in rows: print(f"  {k}  ms/tok={a:6.2f}  mean_accept_len={b:.2f}")
missing=[k for k in sorted(ms) if k not in al]
if missing: print("  NO acceptance data for: "+", ".join(missing))
if len(rows)>=3:
    xs=[r[1] for r in rows]; ys=[r[2] for r in rows]
    mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
    cov=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=sum((x-mx)**2 for x in xs)**.5; dy=sum((y-my)**2 for y in ys)**.5
    r=cov/(dx*dy) if dx and dy else 0.0
    print(f"  n={len(rows)}  ms/tok {min(xs):.1f}-{max(xs):.1f} ({max(xs)/min(xs):.2f}x)"
          f"  accept_len {min(ys):.2f}-{max(ys):.2f}  pearson r={r:+.3f}")
    if max(xs)/min(xs) < 1.25:
        print("  INCONCLUSIVE: these starts did not reproduce the spread; correlation untestable")
    elif r < -0.7: print("  VERDICT: acceptance IS the channel (strong negative correlation)")
    elif abs(r) < 0.4: print("  VERDICT: acceptance is NOT sufficient -- cost lies elsewhere")
    else: print("  VERDICT: ambiguous, more starts needed")
else:
    print("  too few paired rows to correlate")
