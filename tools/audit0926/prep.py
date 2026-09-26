import sqlite3, re, collections, sys
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
ev=[(s,e,S[n],g,S[d]) for s,e,n,g,d in c.execute("select start,end,shortName,graphNodeId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL")]
ev+=[(s,e,'memcpy k%d %dB'%(ck,b),g,'') for s,e,ck,b,g in c.execute("select start,end,copyKind,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev+=[(s,e,'memset %dB'%b,g,'') for s,e,b,g in c.execute("select start,end,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET")]
ev.sort()
heads=[i for i,x in enumerate(ev) if x[2]=='_nvfp4_rows_gemv_kernel']
i=heads[len(heads)//2]
# find a third head: next head followed by a graph within 200 ops and then many graphs
for h in heads[len(heads)//2:]:
    j=h+1
    while ev[j][3] is None: j+=1
    # target if the following graph run is followed by gdn verify soon
    if any('recoverssm_verify' in x[2] for x in ev[j:j+200]) and not any('_nvfp4_rows' in x[2] for x in ev[j:j+200]):
        break
seg=ev[h+1:j]
def short(x):
    d=x[4]; m=re.search(r"(direct_copy|FillFunctor|CUDAFunctor_add<[^>]*>|CUDAFunctorOnSelf_add<[^>]*>|MulFunctor|BinaryFunctor<[^,]*, [^,]*, [^,]*, [^>]*>|OpaqueType<\(int\)\d>|ArgMax|MeanOps|compare_scalar|where|masked|arange|clamp|div_floor|index_put|scatter|gather|copy_kernel|CompareEq|ne_kernel|lt_kernel|ge_kernel|sub_kernel|cumsum|remainder|fmod)", d)
    return x[2] + (':'+m.group(1)[:60] if m else '')
print(len(seg), "ops;", (ev[j][0]-ev[h][1])/1000, "us")
for x in seg: print(f"{(x[1]-x[0])/1000:5.1f} {short(x)[:120]}")
