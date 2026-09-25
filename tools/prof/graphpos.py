import sys, re, collections, statistics as st
S = sys.argv[1]
A = [l.rstrip("\n").split("\t") for l in open(f"{S}/w/prof/cpuops.tsv") if l.startswith("user_annotation")]
gen = sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo, hi = gen[5], gen[-1]; nsteps = len(gen) - 6
R = {}
for l in open(f"{S}/w/prof/runtime.tsv"):
    c, ts, d, ext, corr, name = l.rstrip("\n").split("\t"); R[corr] = name
grid = {}
for l in open(f"{S}/w/kernels2.tsv"):
    c, ts, d, s, g, b, nm = l.rstrip("\n").split("\t", 6); grid[round(float(ts), 1)] = g
def short(nm):
    m = re.search(r"(fp8_blockwise|GemmUniversal|wmma_tensorop_bf16\w*?128x\d|gemvx|_[a-z_]+_kernel|vectorized_elementwise|elementwise_kernel|act_and_mul|splitKreduce|persistent_topk|per_token_group_quant|cvt_fp16_to_fp4|nvfp4_rows_gemv|topkGating|fused_sigmoid_gating\w*)", nm)
    return m.group(1) if m else nm[:30]
K = []
for l in open(f"{S}/w/prof/kernels.tsv"):
    c, ts, d, ext, corr, s, nm = l.rstrip("\n").split("\t")
    t = float(ts)
    if c == "kernel" and s == "33" and lo - 1e5 <= t < hi: K.append((t, float(d), corr, short(nm), grid.get(round(t, 1), "?")))
K.sort(); cnt = collections.Counter(); P = []
for t, d, corr, nm, g in K:
    p = cnt[corr] if R.get(corr) == "cudaGraphLaunch" else -1
    if R.get(corr) == "cudaGraphLaunch": cnt[corr] += 1
    if lo <= t < hi: P.append((p, d, nm, g))
by = collections.defaultdict(lambda: collections.defaultdict(list))
for p, d, nm, g in P:
    cls = "eager" if p < 0 else "p0" if p == 0 else "p1-3" if p <= 3 else "p4+"
    by[(nm, g)][cls].append(d)
print(f"{'kernel':34s} {'grid':12s} {'p0':>14s} {'p1-3':>14s} {'p4+':>14s} {'eager':>14s}   excess/step")
tot = 0
rows = []
for (nm, g), d in by.items():
    base = st.median(d["p4+"]) if len(d["p4+"]) >= 20 else (st.median(d["eager"]) if len(d["eager"]) >= 20 else None)
    if base is None: continue
    ex = sum(max(0, x - base) for c in ("p0", "p1-3") for x in d[c]) / nsteps
    rows.append((ex, nm, g, d, base))
for ex, nm, g, d, base in sorted(rows, key=lambda r: -r[0])[:16]:
    f = lambda c: f"{st.median(d[c]):6.1f}/{len(d[c])/nsteps:4.1f}" if d[c] else "      -     "
    print(f"{nm[:34]:34s} {g:12s} {f('p0'):>14s} {f('p1-3'):>14s} {f('p4+'):>14s} {f('eager'):>14s}   {ex/1e3:6.3f} ms")
    tot += ex
print(f"total excess of p0..p3 over the same kernel at p4+: {tot/1e3:.2f} ms/step  (median us / count per step)")
