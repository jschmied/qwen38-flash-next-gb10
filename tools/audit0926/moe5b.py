import sys, re, collections, sqlite3, statistics as st
sys.path.insert(0,'/home/jschmied/git/qwen38-flash-next-gb10/tools/prof')
import nsyscmp
p=sys.argv[1]
c=sqlite3.connect(p)
S=dict(c.execute("select id,value from StringIds"))
W,steps,span=nsyscmp.load(p)
lo=W[0][0]; hi=W[-1][0]
rows=c.execute("select start,end,shortName,demangledName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<=? order by start",(lo,hi)).fetchall()
idx={id(r):i for i,r in enumerate(rows)}
bystream=collections.defaultdict(list)
for r in rows: bystream[r[7]].append(r)
res=collections.defaultdict(list); G1=[];G2=[]
shared_k=collections.defaultdict(list)
for s,L in bystream.items():
    for i,r in enumerate(L):
        if S[r[2]]!='topkGating': continue
        # routed chain
        j=i
        while not S[L[j][2]].startswith('finalize'): j+=1
        seq=L[i:j+1]
        ex=[q for q in seq if S[q[2]]=='expandInputRowsKernel'][0]
        if ex[4]!=40: continue
        # router gemm: kernels before topk on same stream (wmma 8,4,8 + splitK)
        k=i-1; rstart=L[i][0]
        while k>=0 and (S[L[k][2]] in('Kernel2','splitKreduce_kernel')) and L[k][0]>L[i][0]-80000:
            rstart=L[k][0]; 
            if S[L[k][2]]=='Kernel2': res['router_us'].append((L[k][1]-L[k][0])/1000); break
            k-=1
        fin_end=seq[-1][1]
        gg=[q for q in seq if 'GroupProblemShape' in S[q[3]]]
        G1.append((gg[0][1]-gg[0][0])/1000); G2.append((gg[1][1]-gg[1][0])/1000)
        # other-stream kernels between rstart-60us and fin_end
        oth=[q for q in rows if q[7]!=s and q[0]>=rstart-60000 and q[0]<fin_end]
        # shared expert chain: from wmma(8,10,1) through elementwise; take those starting after rstart-60us
        sh=[q for q in oth if (S[q[2]]=='Kernel2' and (q[4],q[5],q[6]) in((8,10,1),(8,20,1))) or S[q[2]] in('act_and_mul_kernel','vectorized_elementwise_kernel','elementwise_kernel') or ('gemv' in S[q[3]] and q[4]==1)]
        if sh:
            res['shared_busy_us'].append(sum(q[1]-q[0] for q in sh)/1000)
            res['shared_end_before_fin_us'].append((fin_end-max(q[1] for q in sh))/1000)
            for q in sh: shared_k[(S[q[2]],q[4],q[5],q[6])].append((q[1]-q[0])/1000)
        res['router_to_fin_wall_us'].append((fin_end-rstart)/1000)
        # next kernel on any stream after fin_end
for k,v in res.items(): print(f"{k:28s} n={len(v)} mean={st.mean(v):7.1f} med={st.median(v):7.1f} min={min(v):7.1f}")
for k,v in sorted(shared_k.items(),key=lambda x:-sum(x[1])): print(k, "n/step=%.1f mean=%.2f ms/step=%.3f"%(len(v)/steps, st.mean(v), sum(v)/steps/1000))

import statistics as st
m1,m2=st.mean(G1),st.mean(G2)
cov=sum((a-m1)*(b-m2) for a,b in zip(G1,G2))/len(G1)
print("corr G1,G2 %.3f"%(cov/(st.pstdev(G1)*st.pstdev(G2))), "median G1/G2 %.2f"%st.median([a/b for a,b in zip(G1,G2)]))
b1=1843200;b2=921600
print("implied E at 220: G1 %.1f G2 %.1f"%(m1*1e-6*220e9/b1, m2*1e-6*220e9/b2))
print("GB/s if 40 experts: G1 %.0f G2 %.0f"%(40*b1/(m1*1e-6)/1e9, 40*b2/(m2*1e-6)/1e9))
print("GB/s at E=26.8: G1 %.0f G2 %.0f"%(26.8*b1/(m1*1e-6)/1e9, 26.8*b2/(m2*1e-6)/1e9))
q=sorted(G1); print("G1 pct", [round(q[int(len(q)*f)],1) for f in (.05,.25,.5,.75,.95)])
