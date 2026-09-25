import sqlite3, statistics as st, collections, re
c = sqlite3.connect("/opt/llm/capture/nsys-0925/fn.sqlite")
S = dict(c.execute("select id, value from StringIds"))
K = [(s, e, S.get(sn, "?"), S.get(dn, "?"), gx, gy, gz, corr, gn, stream) for s, e, sn, dn, gx, gy, gz, corr, gn, stream in
     c.execute("select start,end,shortName,demangledName,gridX,gridY,gridZ,correlationId,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
RT = {corr: (S.get(nid, "?"), s, e) for corr, nid, s, e in c.execute("select correlationId, nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME")}
main = collections.Counter(k[9] for k in K).most_common(1)[0][0]
Km = [k for k in K if k[9] == main]
print("kernels", len(K), "main stream", main, len(Km))
# first/last 30 % trimmed: the window has warm-up requests at the start
lo, hi = Km[len(Km)//10][0], Km[-len(Km)//10][0]
def isfp8(dn): return "fp8_blockwise" in dn
rows = collections.defaultdict(list)
for i, k in enumerate(Km):
    s, e, sn, dn, gx, gy, gz, corr, gn, stream = k
    if not (lo <= s < hi): continue
    d = (e - s) / 1000
    prev = Km[i-1]
    gap = (s - prev[1]) / 1000
    rt = RT.get(corr, ("?", 0, 0))
    if isfp8(dn) and gy == 20 and gx == 1:
        # previous kernels: GDN (fused_sigmoid) within last 12 -> GDN out_proj
        gdn = any("fused_sigmoid" in S.get(Km[i-q][2], "") or "fused_sigmoid" in Km[i-q][2] for q in range(1, 14))
        # same graph launch: kernels sharing corr before this one
        pos = sum(1 for q in range(1, 40) if Km[i-q][7] == corr)
        first = Km[i-pos]
        launch_lag = (first[0] - rt[2]) / 1000          # first kernel of this launch start - host launch API end
        rows["gdn_out_proj" if gdn else "qsa_o_proj"].append((d, gap, pos, rt[0], (rt[2]-rt[1])/1000, launch_lag, (s - first[0])/1000))
for k, v in rows.items():
    print(f"== {k}: n={len(v)} dur median {st.median(x[0] for x in v):.1f} us | gap before {st.median(x[1] for x in v):.2f} us | "
          f"pos in launch {collections.Counter(x[2] for x in v).most_common(3)} | api {collections.Counter(x[3] for x in v).most_common(2)} "
          f"api dur {st.median(x[4] for x in v):.1f} us | first-kernel start - api end {st.median(x[5] for x in v):.1f} us | "
          f"this kernel start - launch first kernel start {st.median(x[6] for x in v):.1f} us")

MC = [(s, e, b, k, st_) for s, e, b, k, st_ in c.execute("select start,end,bytes,copyKind,streamId from CUPTI_ACTIVITY_KIND_MEMCPY order by start")]
MS = [(s, e, b, st_) for s, e, b, st_ in c.execute("select start,end,bytes,streamId from CUPTI_ACTIVITY_KIND_MEMSET order by start")]
import bisect
mcs = [m[0] for m in MC]
def ctx(i, n=10):
    out = []
    for q in range(n, 0, -1):
        k = Km[i-q]; out.append(f"{S.get(k[2], k[2])[:28]}[{k[4]},{k[5]},{k[6]}] {(k[1]-k[0])/1000:.1f}us")
    return out
shown = set()
for i, k in enumerate(Km):
    s, e, sn, dn, gx, gy, gz, corr, gn, stream = k
    if not (lo <= s < hi) or not (isfp8(dn) and gy == 20 and gx == 1): continue
    gdn = any("fused_sigmoid" in S.get(Km[i-q][2], "") for q in range(1, 14))
    key = "gdn" if gdn else "qsa"
    if key in shown: continue
    shown.add(key)
    print(f"\n== context before one {key} out_proj ({(e-s)/1000:.1f} us):")
    for line in ctx(i, 12): print("   ", line)
    w0 = Km[i-12][0]; j = bisect.bisect_left(mcs, w0)
    while j < len(MC) and MC[j][0] < e:
        m = MC[j]; print(f"    memcpy kind={m[3]} bytes={m[2]} dur={(m[1]-m[0])/1000:.1f}us stream={m[4]} at +{(m[0]-w0)/1000:.1f}us"); j += 1
# statistic: DtoD memcpy bytes per GDN layer window
tot = collections.Counter()
for m in MC:
    if lo <= m[0] < hi: tot[(m[3], m[2])] += 1
print("\nmemcpy (kind, bytes) counts in window:", tot.most_common(8))
