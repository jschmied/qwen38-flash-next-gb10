import sqlite3, re, collections, sys
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
spec=[k for k in K if re.search(r"_gdn_recoverssm_verify_kernel|fused_sigmoid_gating", S[k[2]])]
lo,hi=spec[len(spec)//10][0], spec[-len(spec)//10][0]; steps=sum(1 for k in spec if lo<=k[0]<hi)/36
ev=[(s,e,S[n],g) for s,e,n,g,st in K if lo<=s<hi]
ev+=[(s,e,'memcpy%d:%d'%(ck,b),g) for s,e,ck,b,g in c.execute("select start,end,copyKind,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",(lo,hi))]
ev+=[(s,e,'memset:%d'%b,g) for s,e,b,g in c.execute("select start,end,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(lo,hi))]
ev.sort()
# build union timeline with idle gaps; segments = maximal runs of eager ops
segs=[]; cur=None
for x in ev:
    if x[3] is None:
        if cur is None: cur=[x]
        else: cur.append(x)
    else:
        if cur: segs.append(cur); cur=None
if cur: segs.append(cur)
def region(seg):
    names=' '.join(x[2] for x in seg)
    if 'recoverssm_verify' in names or 'causal_conv1d_update' in names: return 'GDN'
    if '_qsa_' in names: return 'QSA'
    if '_commit_gdn_state' in names: return 'commit'
    return 'other'
# idle per region: gap between previous op end (union) and each op start, assigned to region of the segment containing the op, or 'graph' if op is graph
end=None; idle=collections.Counter(); ops=collections.Counter(); busy=collections.Counter()
segid={}
for i,sg in enumerate(segs):
    r=region(sg)
    for x in sg: segid[id(x)]=r
    ops[r]+=len(sg); busy[r]+=sum(x[1]-x[0] for x in sg)
for x in ev:
    r=segid.get(id(x),'graph')
    if end is not None and x[0]>end: idle[r]+=x[0]-end
    end = x[1] if end is None else max(end,x[1])
cnt=collections.Counter(region(s) for s in segs)
print("steps",steps)
for r in ['GDN','QSA','commit','other','graph']:
    print(f"{r:7s} segs/st {cnt[r]/steps:5.1f} eager ops/st {ops[r]/steps:6.1f} eager busy {busy[r]/steps/1e6:6.3f} ms  idle-before-op {idle[r]/steps/1e6:6.3f} ms")
print("total idle", sum(idle.values())/steps/1e6)
# list 'other' segments composition
comp=collections.Counter()
for sg in segs:
    if region(sg) in ('other','commit'):
        for x in sg: comp[re.sub(r':\d+','',x[2])[:45]]+=1
for k,v in comp.most_common(40): print(f"  {v/steps:5.1f} {k}")
