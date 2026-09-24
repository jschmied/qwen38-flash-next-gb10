import collections, bisect, re
K=[l.rstrip("\n").split("\t") for l in open("prof/kernels.tsv")]
A=[l.rstrip("\n").split("\t") for l in open("prof/cpuops.tsv") if l.startswith("user_annotation")]
gen=sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
pre=[float(a[1]) for a in A if not a[5].startswith("execute_context_0(0)_generation_1(4)")]
K=[(float(k[1]),float(k[2]),k[0],k[5],k[6]) for k in K]
K.sort()
# assign kernels to steps: kernel ts in [gen[i], gen[i+1]); last step bounded by +200ms
bounds=gen+[gen[-1]+200e3]
def cat(n):
    if "wmma_tensorop_bf16" in n: return "bf16 wmma gemm (cutlass80)"
    if "gemvx" in n or "gemv" in n.lower() and "nvfp4" not in n: return "cublas gemv"
    if "nvfp4" in n: return "nvfp4 draft head"
    if "blockwise" in n.lower() or "fp8" in n.lower() and "gemm" in n.lower(): return "fp8 blockwise gemm"
    if "GemmUniversal" in n or "cutlass" in n.lower() and "moe" in n.lower(): return "cutlass GemmUniversal"
    if "scaled_mm" in n: return "cutlass_scaled_mm"
    if re.search(r"fused_moe|moe|expert|topk|grouped", n, re.I): return "moe misc"
    if re.search(r"gdn|chunk|recurrent|conv1d|causal_conv|fla|gated_delta|solve|kkt", n, re.I): return "gdn"
    if re.search(r"flash|attn|attention|indexer|mla|paged|qsa|sparse", n, re.I): return "attention/qsa"
    if re.search(r"triton", n, re.I): return "triton other"
    if re.search(r"norm", n, re.I): return "norm"
    if re.search(r"elementwise|reduce|copy|fill|index|cat|scatter|gather|softmax|argmax|sort|Memcpy|Memset", n, re.I): return "aten elementwise/copy"
    return "other"
per=collections.defaultdict(lambda: collections.defaultdict(float)); cnt=collections.defaultdict(lambda: collections.defaultdict(int))
busy=[0.0]*len(gen); span=[0.0]*len(gen); names=collections.defaultdict(float)
j=0
steps=collections.defaultdict(list)
for ts,d,c,st,n in K:
    i=bisect.bisect_right(bounds,ts)-1
    if 0<=i<len(gen): steps[i].append((ts,d,c,st,n))
for i,ks in steps.items():
    ks.sort(); s0=ks[0][0]; e=s0; b=0; end=max(t+d for t,d,*_ in ks)
    for t,d,c,st,n in ks:
        if t+d>e:
            b+= (t+d)-max(t,e); e=t+d
        cc=cat(n); per[i][cc]+=d; cnt[i][cc]+=1
        if cc=="other": names[n[:90]]+=d
    busy[i]=b; span[i]=end-s0
idx=sorted(steps)[5:]   # skip first 5 steps
m=lambda xs: sorted(xs)[len(xs)//2]
print(f"steps used {len(idx)}; GPU span/step median {m([span[i] for i in idx])/1e3:.2f} ms; busy(union) {m([busy[i] for i in idx])/1e3:.2f} ms")
print(f"step start interval median {m([bounds[i+1]-bounds[i] for i in idx[:-1]])/1e3:.2f} ms")
cats=collections.Counter()
for i in idx:
    for c,v in per[i].items(): cats[c]+=v/len(idx)
tot=sum(cats.values())
for c,v in cats.most_common(): print(f"{c:32s} {v/1e3:7.2f} ms/step  n={sum(cnt[i][c] for i in idx)/len(idx):6.0f}")
print("sum of kernel durations", round(tot/1e3,2))
print("\ntop 'other':"); 
for n,v in sorted(names.items(), key=lambda x:-x[1])[:12]: print(f"{v/len(steps)/1e3:6.3f} ms/step {n}")
