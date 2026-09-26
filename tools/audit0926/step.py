import sqlite3, re, collections, sys
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
ev=[(s,e,S[n],g,gx,gy,gz) for s,e,n,g,gx,gy,gz in c.execute("select start,end,shortName,graphNodeId,gridX,gridY,gridZ from CUPTI_ACTIVITY_KIND_KERNEL")]
ev+=[(s,e,'memcpy%d:%d'%(ck,b),g,0,0,0) for s,e,ck,b,g in c.execute("select start,end,copyKind,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev+=[(s,e,'memset:%d'%b,g,0,0,0) for s,e,b,g in c.execute("select start,end,bytes,graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET")]
ev.sort()
cm=[i for i,x in enumerate(ev) if x[2]=='_commit_gdn_state_kernel']
# commits come in 3 per step; pick the step boundaries at first of each triple
firsts=[cm[0]]+[cm[i] for i in range(1,len(cm)) if ev[cm[i]][0]-ev[cm[i-1]][1]>5e6]
a,b=firsts[len(firsts)//2], firsts[len(firsts)//2+1]
out=[]; g=None; prev_end=ev[a][0]; idle=0
for x in ev[a:b]:
    gap=max(0,x[0]-prev_end); prev_end=max(prev_end,x[1])
    if x[3] is not None:
        if g is None: g=[x[0],x[1],0,gap]
        g[1]=max(g[1],x[1]); g[2]+=1
    else:
        if g: out.append(f"  [G {g[2]} k, {(g[1]-g[0])/1000:.0f} us, gap-in {g[3]/1000:.1f}]"); g=None
        out.append(f"{gap/1000:5.1f} {(x[1]-x[0])/1000:6.1f} {x[2][:40]} ({x[4]},{x[5]},{x[6]})")
if g: out.append(f"  [G {g[2]} k, {(g[1]-g[0])/1000:.0f} us]")
print("step span ms", (ev[b][0]-ev[a][0])/1e6)
print("\n".join(out))
