import sqlite3,re,collections,statistics as st
c=sqlite3.connect('nsys0926/rssm.sqlite')
S=dict(c.execute("select id,value from StringIds"))
K=[(s,e,S[sn],S[dn],gx,gy,gz,stream) for s,e,sn,dn,gx,gy,gz,stream in c.execute("select start,end,shortName,demangledName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
spec=[k for k in K if re.search(r"_gdn_recoverssm_verify_kernel",k[2])]
lo,hi=spec[len(spec)//10][0],spec[-len(spec)//10][0]
m=[k for k in K if lo<=k[0]<hi and k[7]==33]; steps=48.0
# HC site span: from norm kernel start to gate_mix end
span=[];glue=[];gemm=[]
for i,k in enumerate(m):
    if k[2] in('_hc_combine_norm_kernel','_grouped_gemma_rmsnorm_kernel'):
        j=i
        while j<len(m) and m[j][2]!='_hc_gate_mix_kernel' and j<i+8: j+=1
        if m[j][2]!='_hc_gate_mix_kernel': continue
        seq=m[i:j+1]
        tot=(seq[-1][1]-seq[0][0])/1000
        g=sum((x[1]-x[0])/1000 for x in seq if x[2]=='Kernel2')
        span.append(tot); gemm.append(g)
        glue.append(tot-g)
print('sites/step',len(span)/steps,'span med',st.median(span),'gemm med',st.median(gemm),'glue+gaps med',st.median(glue),'ms/step glue+gaps',sum(glue)/steps/1000,'span ms/step',sum(span)/steps/1000)
seen=set()
for k in m:
    if k[2] in ('kernel',) and k[4]==1 and k[5]==1 and k[3][:40] not in seen:
        seen.add(k[3][:40]); print('1x1 kernel:',k[3][:200])
for g in [(8,10),(8,4),(8,3)]:
    for k in K:
        if k[2]=='Kernel2' and (k[4],k[5])==g: print(g,k[3][:160]); break
