import sys, sqlite3, re, collections
sys.path.insert(0, '/home/jschmied/git/qwen38-flash-next-gb10/tools/prof')
db = sys.argv[1]
c = sqlite3.connect(db)
S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,demangledName,gridX,gridY,gridZ,blockX,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
pat = r"fused_sigmoid_gating_delta_rule_update|_gdn_recoverssm_verify_kernel"
spec = [k for k in K if re.search(pat, S[k[2]])]
lo, hi = spec[len(spec)//10][0], spec[-len(spec)//10][0]
steps = sum(1 for k in spec if lo <= k[0] < hi)/36
W = [k for k in K if lo <= k[0] < hi]
print("steps", steps, "span ms/step", (hi-lo)/1e6/steps)
agg = collections.defaultdict(lambda: [0,0.0,0,set()])
for s,e,sn,dn,gx,gy,gz,bx,st,gn,cid in W:
    name = S[sn]
    d = S[dn]
    key = name if name not in ("elementwise_kernel","vectorized_elementwise_kernel","unrolled_elementwise_kernel","kernel","device_kernel","Kernel2","index_elementwise_kernel","reduce_kernel") else name+" :: "+re.sub(r"\s+"," ",d)[:160]
    a = agg[key]; a[0]+=1; a[1]+=(e-s)/1e3; a[2]+= (gn is None); a[3].add((gx,gy,gz))
rows = sorted(agg.items(), key=lambda x:-x[1][1])
for k,(n,t,eag,grids) in rows:
    if t/steps < 5: continue
    print(f"{t/steps/1000:7.3f} ms {n/steps:7.1f}/st eager={eag/steps:5.1f} g={list(grids)[:3]} | {k[:230]}")
