import sqlite3,re,collections,statistics as st,bisect
c=sqlite3.connect('nsys0926/rssm.sqlite')
S=dict(c.execute("select id,value from StringIds"))
K=[(s,e,S[sn],gx,gy,gz,stream) for s,e,sn,gx,gy,gz,stream in c.execute("select start,end,shortName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
spec=[k for k in K if k[2]=='_gdn_recoverssm_verify_kernel']
lo,hi=spec[len(spec)//10][0],spec[-len(spec)//10][0]
W=[k for k in K if lo<=k[0]<hi]
oth=[k for k in W if k[6]!=33]; oe=[k[0] for k in oth]
def ovl(k):
    t=0; j=max(0,bisect.bisect_left(oe,k[0]-3_000_000))
    for o in oth[j:]:
        if o[0]>k[1]: break
        t+=max(0,min(o[1],k[1])-max(o[0],k[0]))
    return t/(k[1]-k[0])
agg=collections.defaultdict(lambda:[0,0])
for k in W:
    if k[6]!=33: continue
    key=k[2] if k[2]!='Kernel2' else f"K2 {k[3]}x{k[4]}x{k[5]}"
    if key.startswith('_hc') or key.startswith('K2') or key=='splitKreduce_kernel':
        f=ovl(k); d=(k[1]-k[0])/1000; agg[key][0]+=d; agg[key][1]+=d*f
for k,(d,h) in sorted(agg.items(),key=lambda x:-x[1][0]): print(f"{k:28s} ms/step {d/48/1000:.3f} hidden-frac {h/d:.2f}")
