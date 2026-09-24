import collections,bisect,re
A=[l.rstrip("\n").split("\t") for l in open("prof/cpuops.tsv") if l.startswith("user_annotation")]
gen=sorted(float(a[1]) for a in A if a[5].startswith("execute_context_0(0)_generation_1(4)"))
lo,hi=gen[5],gen[-1]; n=len(gen)-1-5
K=[]
for l in open("prof/kernels.tsv"):
    c,ts,d,ext,corr,s,name=l.rstrip("\n").split("\t")
    t=float(ts)
    if lo<=t<hi: K.append((t,t+float(d),s,name))
side=sorted((a,b) for a,b,s,_ in K if s!="33")
# merge side intervals
M=[]
for a,b in side:
    if M and a<=M[-1][1]: M[-1][1]=max(M[-1][1],b)
    else: M.append([a,b])
starts=[m[0] for m in M]
def ov(a,b):
    i=max(bisect.bisect_right(starts,a)-1,0); o=0
    while i<len(M) and M[i][0]<b:
        o+=max(0,min(b,M[i][1])-max(a,M[i][0])); i+=1
    return o
def cat(nm):
    if "wmma" in nm: return "bf16 wmma"
    if "gemv" in nm and "nvfp4" not in nm: return "bf16 gemv"
    if "fp8_blockwise" in nm: return "fp8 blockwise"
    if "GemmUniversal" in nm: return "moe grouped gemm"
    if "fused_sigmoid_gating_delta_rule" in nm: return "gdn recurrent update"
    return "other"
ex=collections.Counter(); ovl=collections.Counter()
for a,b,s,nm in K:
    if s!="33": continue
    o=ov(a,b); c=cat(nm); ovl[c]+=o; ex[c]+=(b-a-o)
sidetot=sum(b-a for a,b in M)
print(f"side-stream busy (merged) {sidetot/n/1e3:.2f} ms/step")
for c in ex: print(f"{c:22s} exclusive {ex[c]/n/1e3:6.2f}  overlapped-with-side {ovl[c]/n/1e3:6.2f} ms/step")
