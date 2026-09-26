import sqlite3, re, collections, sys, statistics as st
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
ev=[(s,e,S[n],g,gx) for s,e,n,g,gx in c.execute("select start,end,shortName,graphNodeId,gridY from CUPTI_ACTIVITY_KIND_KERNEL")]
ev+=[(s,e,'memcpy',g,0) for s,e,g in c.execute("select start,end,graphNodeId from CUPTI_ACTIVITY_KIND_MEMCPY")]
ev+=[(s,e,'memset',g,0) for s,e,g in c.execute("select start,end,graphNodeId from CUPTI_ACTIVITY_KIND_MEMSET")]
ev.sort()
cm=[i for i,x in enumerate(ev) if x[2]=='_commit_gdn_state_kernel']
firsts=[cm[0]]+[cm[i] for i in range(1,len(cm)) if ev[cm[i]][0]-ev[cm[i-1]][1]>5e6]
res=collections.defaultdict(list)
for a,b in zip(firsts[3:-4], firsts[4:-3]):
    E=ev[a:b]
    heads=[i for i,x in enumerate(E) if x[2]=='_nvfp4_rows_gemv_kernel']
    lmh=[i for i,x in enumerate(E) if x[2]=='device_kernel' and x[3] is None and x[4]==1940]
    firstg=[i for i,x in enumerate(E) if x[3] is not None]
    if len(heads)!=3 or len(lmh)!=1: continue
    tgt0=[i for i in firstg if i>heads[2]][0]
    d1g=firstg[0]
    bounds=[0,d1g,heads[0]+2,heads[1]+2,heads[2]+2,tgt0,lmh[0],len(E)]
    names=['commit+post','drafter1-prep? (pre graph)','drafter step1 (M=4, graphs+QSA eager)','drafter step2 (M=1)','drafter step3 (M=1)','target input prep','target forward','lm_head+rejection']
    # fix naming: segment0 commit..first graph
    names=['commit+postproc+drafter1 prep','drafter step1 fwd+head','drafter step2 (prep+fwd+head)','drafter step3 (prep+fwd+head)','target input prep (after head3)','target fwd (graphs+eager attn)','lm_head+rejection']
    bounds=[0,d1g,heads[0]+2,heads[1]+2,heads[2]+2,tgt0,lmh[0],len(E)]
    bounds=[0]+bounds[1:]
    bounds=[bounds[0],bounds[1],bounds[2],bounds[3],bounds[4],bounds[5],bounds[6],bounds[7]]
    for k in range(7):
        seg=E[bounds[k]:bounds[k+1]]
        if not seg: continue
        t0=seg[0][0]; t1=(E[bounds[k+1]][0] if bounds[k+1]<len(E) else ev[b][0])
        busy=0; end=None
        for x in seg:
            s,e=x[0],min(x[1],t1)
            if end is None or s>end: busy+=e-s; end=e
            elif e>end: busy+=e-end; end=e
        eager=sum(1 for x in seg if x[3] is None)
        res[names[k]].append(((t1-t0)/1e3, busy/1e3, eager, len(seg)))
tot=[0,0,0]
for n in ['commit+postproc+drafter1 prep','drafter step1 fwd+head','drafter step2 (prep+fwd+head)','drafter step3 (prep+fwd+head)','target input prep (after head3)','target fwd (graphs+eager attn)','lm_head+rejection']:
    v=res[n]; sp=st.median(x[0] for x in v); bu=st.median(x[1] for x in v); eg=st.median(x[2] for x in v); al=st.median(x[3] for x in v)
    tot[0]+=sp; tot[1]+=bu; tot[2]+=eg
    print(f"{n:34s} n={len(v)} span {sp:8.1f} us busy {bu:8.1f} idle {sp-bu:6.1f} eager ops {eg:5.0f} / {al:5.0f}")
print("sum", [round(x,1) for x in tot])
