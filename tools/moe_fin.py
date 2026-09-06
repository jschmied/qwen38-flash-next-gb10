# Per-kernel times of one FlashInfer fused-MoE layer call: fused finalize vs unfused, via torch profiler. Usage: moe_fin.py M
import torch, sys, collections
import vllm  # noqa
from vllm import _custom_ops as ops
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
from torch.profiler import profile, ProfilerActivity
dev="cuda"; torch.manual_seed(0); E,H,I,TOPK=512,2560,640,10; M=int(sys.argv[1]) if len(sys.argv)>1 else 7503
w13=torch.randint(0,256,(E,2*I,H//2),device=dev,dtype=torch.uint8); w2=torch.randint(0,256,(E,H,I//2),device=dev,dtype=torch.uint8)
w13_s=(torch.rand(E,2*I,H//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn); w2_s=(torch.rand(E,H,I//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn)
g1=torch.ones(E,device=dev); g2=torch.ones(E,device=dev); a1g=torch.ones((),device=dev); a2g=torch.ones((),device=dev)
x=torch.randn(M,H,device=dev,dtype=torch.bfloat16); a_fp4,a_sf=ops.scaled_fp4_quant(x,a1g)
topk_ids=torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32); topk_w=torch.softmax(torch.randn(M,TOPK,device=dev),dim=-1).float(); out=torch.empty(M,H,device=dev,dtype=torch.bfloat16)
def call(fused):
    flashinfer_cutlass_fused_moe(input=a_fp4, token_selected_experts=topk_ids, token_final_scales=topk_w, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
        output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2], input_sf=a_sf, output=out, tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu, use_fused_finalize=fused)
for fused in (True, False):
    try:
        for _ in range(3): call(fused)
    except Exception as ex: print(f"use_fused_finalize={fused}: ERROR {str(ex)[:160]}"); continue
    torch.cuda.synchronize(); N=10
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(N): call(fused)
        torch.cuda.synchronize()
    agg=collections.defaultdict(float); cnt=collections.Counter()
    for ev in p.events():
        if ev.device_type.name=="CUDA": agg[ev.name]+=ev.time_range.elapsed_us()/N; cnt[ev.name]+=1
    tot=sum(agg.values())
    print(f"\n=== M={M} use_fused_finalize={fused}: {tot:.0f} us GPU per layer call")
    for n,us in sorted(agg.items(), key=lambda kv:-kv[1])[:8]:
        short="grouped GEMM (GemmUniversal)" if "GemmUniversal" in n else n.split("<")[0].split("::")[-1][:60]
        print(f"  {us:8.1f} us x{cnt[n]//N:<2d} {short}")
