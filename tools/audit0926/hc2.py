import sqlite3,re,collections,statistics as st
c=sqlite3.connect('nsys0926/rssm.sqlite')
S=dict(c.execute("select id,value from StringIds"))
K=[(s,e,S[sn],gx,gy,gz,stream) for s,e,sn,gx,gy,gz,stream in c.execute("select start,end,shortName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
spec=[k for k in K if re.search(r"_gdn_recoverssm_verify_kernel|fused_sigmoid_gating",k[2])]
lo,hi=spec[len(spec)//10][0],spec[-len(spec)//10][0]
W=[k for k in K if lo<=k[0]<hi]; steps=sum(1 for k in spec if lo<=k[0]<hi)/36
print('steps',steps)
main=[k for k in W if k[6]==W[0][6] or True]
# sequence on the HC stream
MS=collections.Counter(k[6] for k in W).most_common(3); print(MS)
m=[k for k in W if k[6]==MS[0][0]]
def d(k): return (k[1]-k[0])/1000
def nm(k): return f"{k[2]}{k[3]}x{k[4]}x{k[5]}"
# classify mixer-down sites by the kernel two before (norm kernel's predecessor)
sites=collections.defaultdict(list); hcbucket=collections.Counter(); hcn=collections.Counter()
for i,k in enumerate(m):
    if k[2]=='Kernel2' and (k[3],k[4])==(8,3):
        pre=m[i-1]; pp=m[i-2]
        key=(pre[2], pp[2]+f"{pp[3]}x{pp[4]}")
        sites[key].append(d(k))
    if k[2].startswith('_hc_') or k[2]=='_grouped_gemma_rmsnorm_kernel' or (k[2]=='splitKreduce_kernel' and m[i-1][2]=='Kernel2' and (m[i-1][3],m[i-1][4])==(8,3)):
        hcbucket[k[2]]+=d(k); hcn[k[2]]+=1
for key,v in sorted(sites.items(),key=lambda x:-len(x[1])):
    print('mixdown',key,len(v)/steps,'med',round(st.median(v),1))
for g in [(8,3),(8,80),(8,10),(8,4),(8,20),(8,1)]:
    v=[d(k) for k in W if k[2]=='Kernel2' and (k[3],k[4])==g]
    print('K2',g,'calls/step',round(len(v)/steps,1),'med',round(st.median(v),1),'ms/step',round(sum(v)/steps/1000,3))
for n in hcbucket: print(n,'calls/step',round(hcn[n]/steps,1),'ms/step',round(hcbucket[n]/steps/1000,3),'med',)
# 8x10 overlap analysis
oth=[k for k in W if k[6]!=MS[0][0]]
import bisect
os_=[k[0] for k in oth]
res=[]
for i,k in enumerate(m):
    if k[2]=='Kernel2' and (k[3],k[4])==(8,10):
        # overlap with other-stream kernels
        ov=[];
        j=bisect.bisect_left(os_,k[0]-2_000_000)
        for o in oth[j:]:
            if o[0]>k[1]: break
            a=max(o[0],k[0]); b=min(o[1],k[1])
            if b>a: ov.append((nm(o),(b-a)/1000))
        rt=sum(x[1] for x in ov if x[0].startswith('Kernel28x4'))
        res.append((d(k),rt,ov,nm(m[i-1]),nm(m[i-2])))
print('8x10 n',len(res),'dur med',st.median(r[0] for r in res),'router-overlap med',st.median(r[1] for r in res))
print('frac w/ router overlap',sum(1 for r in res if r[1]>0)/len(res))
nov=[r[0] for r in res if r[1]==0]; print('no-overlap durs',nov[:10])
ovk=collections.Counter(); 
for r in res:
    for n_,t in r[2]: ovk[n_]+=t
print({k:round(v/len(res),1) for k,v in ovk.most_common(8)})
print(collections.Counter((r[3],r[4]) for r in res).most_common(3))
# shared down 8x20 overlap, and router alone
