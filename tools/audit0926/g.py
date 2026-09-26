import sqlite3, re, collections, sys
db = sys.argv[1]
c = sqlite3.connect(db)
S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
pat = r"fused_sigmoid_gating_delta_rule_update|_gdn_recoverssm_verify_kernel"
spec = [k for k in K if re.search(pat, S[k[2]])]
lo, hi = spec[len(spec)//10][0], spec[-len(spec)//10][0]
steps = sum(1 for k in spec if lo <= k[0] < hi)/36
RT = {cid:(S.get(n,'?'),s,e) for s,e,n,cid in c.execute("select start,end,nameId,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME")}
ev = [(s,e,S[n][:40],st,gn,cid,'K') for s,e,n,st,gn,cid in K if lo<=s<hi]
ev += [(s,e,'memcpy',st,gn,cid,'M') for s,e,st,gn,cid in c.execute("select start,end,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?",(lo,hi))]
ev += [(s,e,'memset',st,gn,cid,'S') for s,e,st,gn,cid in c.execute("select start,end,streamId,graphNodeId,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and start<?",(lo,hi))]
ev.sort()
gaps = collections.Counter(); gn_ = collections.Counter(); hostb = collections.Counter(); total=0; hb_total=0
bycls = collections.Counter()
end = ev[0][1]; last = ev[0]
for x in ev[1:]:
    s,e,nm,st,gn,cid,t = x
    if s > end + 1000:
        g = s-end
        api = RT.get(cid)
        # host-bound: launch API ended less than 5us before the kernel start AND api started after prev end
        hbound = api is not None and api[2] > end - 0 and (s - api[2]) < 8000
        k = (last[2]+('[G]' if last[4] else '[E]'), nm+('[G]' if gn else '[E]'))
        gaps[k]+=g; gn_[k]+=1; total+=g
        if hbound: hostb[k]+=g; hb_total+=g
        cls = ('graph->graph' if last[4] and gn else 'eager->eager' if not last[4] and not gn else 'graph->eager' if last[4] else 'eager->graph')
        bycls[(cls, 'host' if hbound else 'other', (api[0] if api else '?')[:20])]+=g
    if x[1] > end: end, last = x[1], x
print(f"steps {steps} idle(gaps>1us) {total/steps/1e6:.3f} ms/step; host-bound (launch API ended after prev GPU end) {hb_total/steps/1e6:.3f}")
for k,v in sorted(bycls.items(), key=lambda x:-x[1]): print(f"  {v/steps/1e6:6.3f}  {k}")
for k,v in gaps.most_common(25):
    print(f"{v/steps/1e6:6.3f} ms/st n/st={gn_[k]/steps:5.1f} hostbound={hostb[k]/max(v,1):4.2f}  {k[0]} -> {k[1]}")
