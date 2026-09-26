import sqlite3, re, sys
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
ev = [(s,e,S[n],S[d],g,gx,gy,gz,bx,st,'K') for s,e,n,d,g,gx,gy,gz,bx,st in c.execute("select start,end,shortName,demangledName,graphNodeId,gridX,gridY,gridZ,blockX,streamId from CUPTI_ACTIVITY_KIND_KERNEL")]
ev += [(s,e,'memcpy k%d'%k,str(b),g,0,0,0,0,st,'M') for s,e,k,b,g,st in c.execute("select start,end,copyKind,bytes,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev += [(s,e,'memset','%d'%b,g,0,0,0,0,st,'S') for s,e,b,g,st in c.execute("select start,end,bytes,graphNodeId,streamId from CUPTI_ACTIVITY_KIND_MEMSET")]
ev.sort()
spec=[i for i,x in enumerate(ev) if sys.argv[4] in x[2]]
i0 = spec[len(spec)//2]
a,b = int(sys.argv[2]), int(sys.argv[3])
prev=None
for x in ev[i0+a:i0+b]:
    s,e,n,d,g,gx,gy,gz,bx,st,t=x
    gap = (s-prev)/1000 if prev else 0
    dd = d if t!='K' else re.sub(r"\s+"," ",d)
    m = re.search(r"(direct_copy|FillFunctor|CUDAFunctor_add|MulFunctor|sigmoid|bfloat16_copy|float_copy|BinaryFunctor<[^,]*|index_kernel_impl<[^>]*>|ArgMax|Mean|pow|rsqrt|div_floor|CatArray|gemvx|wmma[^>]*|fp8_blockwise|GroupProblemShape)", dd)
    print(f"{gap:6.1f} {(e-s)/1000:6.1f} {'G' if g else 'E'} s{st} {n[:40]:40s} g=({gx},{gy},{gz})x{bx} {m.group(1)[:40] if m else ''} {dd[:20] if t!='K' else ''}")
    prev=e
