# FlashInfer CUTLASS fused MoE (NVFP4 W4A4) at Flash-Next's expert geometry, prefill M: autotuner pick vs explicit tactic sweep.
import torch, statistics, sys, itertools
import vllm  # noqa
from vllm import _custom_ops as ops
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
dev="cuda"; torch.manual_seed(0)
E,H,I,TOPK=512,2560,640,10
def timeit(fn,n=5,reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)
w13=torch.randint(0,256,(E,2*I,H//2),device=dev,dtype=torch.uint8); w2=torch.randint(0,256,(E,H,I//2),device=dev,dtype=torch.uint8)
w13_s=(torch.rand(E,2*I,H//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn); w2_s=(torch.rand(E,H,I//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn)
g1=torch.ones(E,device=dev,dtype=torch.float32); g2=torch.ones(E,device=dev,dtype=torch.float32); a1g=torch.ones((),device=dev); a2g=torch.ones((),device=dev)
def run(M, profile_ids=None, tune_max=8192):
    x=torch.randn(M,H,device=dev,dtype=torch.bfloat16)
    a_fp4,a_sf=ops.scaled_fp4_quant(x, a1g)
    topk_ids=torch.stack([torch.randperm(E,device=dev)[:TOPK] for _ in range(M)]).to(torch.int32) if M<=512 else torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32)
    topk_w=torch.softmax(torch.randn(M,TOPK,device=dev),dim=-1).to(torch.float32)
    out=torch.empty(M,H,device=dev,dtype=torch.bfloat16)
    def f():
        flashinfer_cutlass_fused_moe(input=a_fp4, token_selected_experts=topk_ids, token_final_scales=topk_w, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
            output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2], input_sf=a_sf, output=out, tune_max_num_tokens=tune_max,
            activation_type=ActivationType.Swiglu, use_fused_finalize=True, profile_ids=profile_ids)
    return f
M=int(sys.argv[1]) if len(sys.argv)>1 else 7503
fl=2*M*TOPK*(3*H*I)  # gate+up+down
print(f"M={M} top-{TOPK}: {fl/1e9:.0f} GFLOP per MoE layer")
f=run(M); t=timeit(f); print(f"  default (heuristic/autotuner cache): {t:9.1f} us  {fl/t/1e6:6.1f} TFLOPS", flush=True)
res=[]
for t1 in range(0,32):
    for t2 in (None,):
        try:
            g=run(M, profile_ids=[t1, -1]); tt=timeit(g,n=3,reps=3); res.append((tt,t1,-1)); print(f"  gemm1 tactic {t1:2d}: {tt:9.1f} us ({fl/tt/1e6:5.1f} TF)", flush=True)
        except Exception as ex: print(f"  gemm1 tactic {t1:2d}: ERR {str(ex)[:80]}", flush=True); break
best1=min(res)[1] if res else -1
res2=[]
for t2 in range(0,64):
    try:
        g=run(M, profile_ids=[best1, t2]); tt=timeit(g,n=3,reps=3); res2.append((tt,best1,t2)); print(f"  gemm1 {best1} + gemm2 tactic {t2:2d}: {tt:9.1f} us ({fl/tt/1e6:5.1f} TF)", flush=True)
    except Exception as ex: print(f"  gemm2 tactic {t2:2d}: ERR {str(ex)[:80]}", flush=True); break
if res2: print("BEST:", min(res2))
