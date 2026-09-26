import pickle,re,collections,statistics as stt
K=pickle.load(open('K.pkl','rb'))
def cls(sn,dn,gx,gy,gz,st):
    n=sn+' '+dn
    if sn=='_nvfp4_rows_gemv_kernel': return 'draft_head'
    if 'fp8_blockwise' in dn and gy==1940: return 'lm_head'
    if re.search(r'_qsa_|reshape_and_cache|persistent_topk|_expand_qsa',n): return 'attn'
    if re.search(r'GemmUniversal',dn): return 'moe_gemm'
    if re.search(r'topkGating|fp16_to_fp4|ExpertPrefix|expandInputRows|computeStrides|doActivation|finalizeMoe',n): return 'moe_route'
    if re.search(r'gdn_recoverssm|fused_sigmoid|_causal_conv1d|fused_gdn|l2norm|_ple_',n): return 'gdn'
    if re.search(r'commit_gdn|prepare_commit|compact_conv|postprocess_recoverssm|postprocess_mamba',n): return 'commit'
    if re.search(r'fp8_blockwise|per_token_group_quant',n): return 'fp8_gemm'
    if re.search(r'Kernel2|gemvx|gemv|splitKreduce|cublas|nvjet|wmma',n): return 'bf16_gemm'
    if re.search(r'_hc_',n): return 'hc'
    if re.search(r'_rejection|_resample|_insert_resampled|_get_num_sampled|_post_update|_scatter_num_accepted|logits_stats',n): return 'sample'
    return 'other'
# cycle boundaries: lm_head kernels
L=[i for i,k in enumerate(K) if 'fp8_blockwise' in k[3] and k[5]==1940]
print('lm_heads',len(L))
rows=[]
for a,b in zip(L[:-1],L[1:]):
    seg=K[a:b+1]
    # markers
    idx={}
    for j in range(a,b+1):
        sn=K[j][2]
        if sn=='_prepare_commit_plan_kernel' and 'c0' not in idx: idx['c0']=j
        if sn=='postprocess_mamba_fused_kernel' and 'c1' not in idx: idx['c1']=j
        if sn=='_prepare_decode_inputs_kernel' and 'd2' not in idx: idx['d2']=j
        if sn=='_update_draft_inputs_kernel': idx.setdefault('u',[]).append(j)
    if len(idx.get('u',[]))!=2 or 'd2' not in idx or 'c0' not in idx: continue
    bounds=[('sample',a+1,idx['c0']),('commit',idx['c0'],idx['c1']+1),('d1',idx['c1']+1,idx['d2']),('d2',idx['d2'],idx['u'][0]),('d3',idx['u'][0],idx['u'][1]+1),('target',idx['u'][1]+1,b+1)]
    r={'wall':(K[b][1]-K[a][1])/1e3}
    for name,s,e in bounds:
        ks=K[s:e]
        t0=ks[0][0]; t1=K[e][0] if e<len(K) and name!='target' else ks[-1][1]
        if name=='target': t0=K[s][0]; t1=K[b][1]
        busy=0;end=None
        for k in sorted(ks):
            if end is None or k[0]>end: busy+=k[1]-k[0]; end=k[1]
            elif k[1]>end: busy+=k[1]-end; end=k[1]
        r[name+'_wall']=(t1-t0)/1e3; r[name+'_busy']=busy/1e3
        bc=collections.Counter()
        for k in ks: bc[cls(k[2],k[3],k[4],k[5],k[6],k[8])]+= (k[1]-k[0])/1e3
        r[name+'_cls']=bc
    rows.append(r)
n=len(rows); lo,hi=n//10,n-n//10; R=rows[lo:hi]
print('cycles',n,'used',len(R))
print('wall/cycle %.2f'%stt.mean(r['wall'] for r in R))
for name in ['target','sample','commit','d1','d2','d3']:
    w=stt.mean(r[name+'_wall'] for r in R); bz=stt.mean(r[name+'_busy'] for r in R)
    cc=collections.Counter()
    for r in R: cc.update(r[name+'_cls'])
    print(f"{name:7s} wall {w:7.3f} busy {bz:7.3f} | "+"  ".join(f"{k}:{v/len(R):.3f}" for k,v in cc.most_common()))
