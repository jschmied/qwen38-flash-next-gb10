import sqlite3, re, collections, sys, bisect
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,demangledName,graphNodeId,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
spec=[k for k in K if re.search(r"_gdn_recoverssm_verify_kernel|fused_sigmoid_gating", S[k[2]])]
lo,hi=spec[len(spec)//10][0], spec[-len(spec)//10][0]; steps=sum(1 for k in spec if lo<=k[0]<hi)/36
W=[k for k in K if lo<=k[0]<hi]
W += [(s,e,'memcpy','memcpy k%d'%ck,g,0,0,0,st) for s,e,ck,g,st in c.execute("select start,end,copyKind,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",(lo,hi))]
W += [(s,e,'memset','memset',g,0,0,0,st) for s,e,g,st in c.execute("select start,end,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(lo,hi))]
W.sort()
# exclusive time: portion of [s,e] where no other-stream op overlaps
other = [(k[0],k[1],k[8]) for k in W]
def name(k):
    sn = S.get(k[2],k[2]) if isinstance(k[2],int) else k[2]
    d = S.get(k[3],k[3]) if isinstance(k[3],int) else k[3]
    m = re.search(r"(direct_copy|FillFunctor|CUDAFunctor_add<[^>]*>|MulFunctor|sigmoid|bfloat16_copy|float_copy|BinaryFunctor<[^,]*|OpaqueType<\(int\)\d>|ArgMax|MeanOps|pow_tensor|rsqrt|div_floor|CUDAFunctorOnSelf_add<[^>]*>|gemvx::kernel<int, int, __nv_bfloat16, (?:float|__nv_bfloat16))", d)
    base = sn if sn not in ('elementwise_kernel','vectorized_elementwise_kernel','unrolled_elementwise_kernel','index_elementwise_kernel','reduce_kernel','kernel') else sn+':'+(m.group(1) if m else '?')
    if 'gemvx' in base or 'wmma' in d: base += f" g{k[5]}x{k[6]}x{k[7]}"
    return base
agg=collections.defaultdict(lambda:[0,0.0,0.0,0])
starts=[o[0] for o in other]
for i,k in enumerate(W):
    s,e,st=k[0],k[1],k[8]
    # collect overlapping intervals of other streams
    iv=[]
    j=bisect.bisect_left(starts, s-2_000_000)
    for o in other[j:]:
        if o[0]>=e: break
        if o[2]!=st and o[1]>s: iv.append((max(o[0],s),min(o[1],e)))
    iv.sort(); cov=0; cur=None
    for a,b in iv:
        if cur is None or a>cur[1]:
            if cur: cov+=cur[1]-cur[0]
            cur=[a,b]
        else: cur[1]=max(cur[1],b)
    if cur: cov+=cur[1]-cur[0]
    a=agg[(name(k), 'G' if k[4] else 'E')]; a[0]+=1; a[1]+=(e-s); a[2]+=(e-s-cov)
tot=sum(v[2] for v in agg.values())
print(f"steps {steps}; exclusive total {tot/steps/1e6:.2f} ms/step")
for k,v in sorted(agg.items(), key=lambda x:-x[1][2]):
    if v[2]/steps < 8000 and v[1]/steps<30000: continue
    print(f"{v[0]/steps:6.1f}/st total {v[1]/steps/1e6:6.3f} excl {v[2]/steps/1e6:6.3f}  {k[1]} {k[0][:110]}")
