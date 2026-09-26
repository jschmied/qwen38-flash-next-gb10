import sqlite3, re, sys, statistics as stt
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
spec=[k for k in K if re.search(r"_gdn_recoverssm_verify_kernel|fused_sigmoid_gating", S[k[2]])]
lo,hi=spec[len(spec)//10][0], spec[-len(spec)//10][0]
RT = {cid:(s,e,S[n]) for s,e,n,cid in c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
L=[]; 
for s,e,n,g,cid in K:
    if lo<=s<hi and cid in RT: L.append((s-RT[cid][1])/1000)
L.sort(); q=lambda p:L[int(p*len(L))]
print("kernel start - launch API end (us): p1 %.1f p10 %.1f p50 %.1f p90 %.1f min %.1f"%(q(.01),q(.1),q(.5),q(.9),L[0]))
# per step: where is host relative to GPU at spec kernel
