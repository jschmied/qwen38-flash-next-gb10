import sys, re, collections, sqlite3, statistics as st
sys.path.insert(0,'/home/jschmied/git/qwen38-flash-next-gb10/tools/prof')
import nsyscmp
p=sys.argv[1]
c=sqlite3.connect(p)
S=dict(c.execute("select id,value from StringIds"))
W,steps,span=nsyscmp.load(p)
lo=W[0][0]; hi=W[-1][0]
rows=c.execute("select start,end,shortName,demangledName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and start<=? order by start",(lo,hi)).fetchall()
def nm(r):
    sn=S[r[2]]; d=S[r[3]]
    if sn=='device_kernel' and 'GroupProblemShape' in d:
        m=re.search(r'cute::tuple<cute::C<\(int\)128>, cute::C<\(int\)(\d+)>, cute::C<\(int\)(\d+)>>',d)
        return 'GG_128x%sx%s'%m.groups() if m else 'GG?'
    return sn
bystream=collections.defaultdict(list)
for r in rows: bystream[r[7]].append(r)
calls=[]
for s,L in bystream.items():
    i=0
    while i<len(L):
        if S[L[i][2]]=='topkGating':
            j=i; seq=[]
            while j<len(L):
                seq.append(L[j])
                if S[L[j][2]].startswith('finalizeMoeRouting'): break
                j+=1
            calls.append(seq); i=j+1
        else: i+=1
print("calls",len(calls),"per step",len(calls)/steps)
agg=collections.defaultdict(list)
for seq in calls:
    ex=[r for r in seq if S[r[2]]=='expandInputRowsKernel']
    M=ex[0][4]//10 if ex else None
    gg=[r for r in seq if nm(r).startswith('GG')]
    t0=seq[0][0]; t1=seq[-1][1]
    busy=sum(r[1]-r[0] for r in seq)
    agg[(M,'wall')].append((t1-t0)/1000); agg[(M,'busy')].append(busy/1000)
    for k,r in enumerate(gg): agg[(M,'G%d:%s'%(k+1,nm(r)))].append((r[1]-r[0])/1000)
    pre=[r for r in seq if r[0]<gg[0][0]]
    agg[(M,'pre_gemm_wall')].append((gg[0][0]-t0)/1000)
    for r in seq:
        n=nm(r)
        if not n.startswith('GG'): agg[(M,n)].append((r[1]-r[0])/1000)
for k in sorted(agg,key=lambda x:(x[0],x[1])):
    v=agg[k]; print(f"M={k[0]} {k[1]:45s} n/step={len(v)/steps:5.1f} mean={st.mean(v):7.2f} med={st.median(v):7.2f} p10={sorted(v)[len(v)//10]:7.2f} p90={sorted(v)[9*len(v)//10]:7.2f}  ms/step={sum(v)/steps/1000:6.3f}")
