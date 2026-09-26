import sqlite3, re, collections, sys
db = sys.argv[1]
c = sqlite3.connect(db)
S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
pat = r"fused_sigmoid_gating_delta_rule_update|_gdn_recoverssm_verify_kernel"
spec = [k for k in K if re.search(pat, S[k[2]])]
lo, hi = spec[len(spec)//10][0], spec[-len(spec)//10][0]
steps = sum(1 for k in spec if lo <= k[0] < hi)/36
RT = {cid:(S.get(n,'?'),s,e,tid,cc) for s,e,n,cid,tid,cc in c.execute("select start,end,nameId,correlationId,globalTid,callchainId from CUPTI_ACTIVITY_KIND_RUNTIME")}
print("steps",steps)
kinds = {0:'?',1:'H2D',2:'D2H',8:'D2D',13:'UVM'}
mk = {r[0]:r[1] for r in c.execute("select id,label from ENUM_CUDA_MEM_KIND")}
agg = collections.defaultdict(lambda:[0,0,0,0])
for s,e,b,ck,sk,dk,st,gn,cid in c.execute("select start,end,bytes,copyKind,srcKind,dstKind,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",(lo,hi)):
    api = RT.get(cid,('?',))[0]
    bb = 'le64' if b<=64 else 'le4K' if b<=4096 else 'le64K' if b<=65536 else 'big'
    key=(kinds.get(ck,ck), mk.get(sk), mk.get(dk), st, 'graph' if gn else 'eager', api, bb)
    a=agg[key]; a[0]+=1; a[1]+=e-s; a[2]+=b
print("MEMCPY: calls/step ms/step bytes/call key")
for k,a in sorted(agg.items(), key=lambda x:-x[1][0]):
    print(f"{a[0]/steps:6.2f} {a[1]/steps/1e6:7.4f} {a[2]/a[0]:10.0f}  {k}")
agg = collections.defaultdict(lambda:[0,0,0])
for s,e,b,v,st,gn,cid,mkd in c.execute("select start,end,bytes,value,streamId,graphNodeId,correlationId,memKind from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(lo,hi)):
    api = RT.get(cid,('?',))[0]
    key=(st,'graph' if gn else 'eager',api,v,b)
    a=agg[key]; a[0]+=1; a[1]+=e-s; a[2]+=b
print("MEMSET")
for k,a in sorted(agg.items(), key=lambda x:-x[1][0]):
    print(f"{a[0]/steps:6.2f} {a[1]/steps/1e6:7.4f} {k}")
print("SYNC")
agg = collections.defaultdict(lambda:[0,0])
for s,e,t,st,cid in c.execute("select start,end,syncType,streamId,correlationId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where start>=? and start<?",(lo,hi)):
    a=agg[(t,st,RT.get(cid,('?',))[0])]; a[0]+=1; a[1]+=e-s
for k,a in sorted(agg.items(), key=lambda x:-x[1][1]):
    print(f"{a[0]/steps:6.2f}/st {a[1]/steps/1e6:7.3f} ms/st {k}")
print("RUNTIME API (window)")
agg = collections.defaultdict(lambda:[0,0])
for s,e,n,tid in c.execute("select start,end,nameId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and start<?",(lo,hi)):
    a=agg[(S.get(n),tid)]; a[0]+=1; a[1]+=e-s
for k,a in sorted(agg.items(), key=lambda x:-x[1][1])[:25]:
    print(f"{a[0]/steps:7.1f}/st {a[1]/steps/1e6:7.3f} ms/st {k}")
print("streams:", collections.Counter(k[3] for k in K if lo<=k[0]<hi))
print("callchains present:", c.execute("select count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where callchainId is not null").fetchone())
