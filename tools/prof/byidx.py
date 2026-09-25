import sys, bisect, collections, statistics as st
def load(D, CPU, MATCH, GRID):
    A = [l.rstrip("\n").split("\t") for l in open(CPU) if l.startswith("user_annotation")]
    gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
    lo, hi = gen[5], gen[-1]; by = collections.defaultdict(list); cnt = collections.Counter()
    for l in open(f"{D}/kernels2.tsv"):
        c, ts, d, s, g, b, n = l.rstrip("\n").split("\t", 6); t = float(ts)
        if lo <= t < hi and s == "33" and MATCH in n and g == GRID:
            stp = bisect.bisect_right(gen, t) - 1; k = cnt[stp]; cnt[stp] += 1; by[k].append(float(d))
    return {k: st.median(v) for k, v in by.items()}, collections.Counter(cnt.values())
S = sys.argv[1]
b, bc = load(f"{S}/base2", f"{S}/prof/cpuops.tsv", "wmma", "[8,3,9]")
k, kc = load(f"{S}/sk2", f"{S}/sk/prof/cpuops.tsv", "_partial", "[21,2,1]")
print("calls/step base", bc.most_common(3), "sk", kc.most_common(3))
line = []
for i in range(max(b)):
    line.append(f"{i:3d}:{b.get(i,0):4.0f}/{k.get(i,0):4.0f}")
for j in range(0, len(line), 8): print("  ".join(line[j:j+8]))
import math
ks = sorted(set(b) & set(k)); xs = [b[i] for i in ks]; ys = [k[i] for i in ks]
mx, my = st.mean(xs), st.mean(ys)
r = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
print("correlation of per-index median, base vs sk:", round(r, 3))
