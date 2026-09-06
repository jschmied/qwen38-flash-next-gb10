# One FlashInfer CUTLASS fused-MoE layer at Flash-Next geometry (512 experts, top-10, 2560x640, NVFP4 W4A4), M rows,
# default (autotuned) tactic. Warmup calls then measured calls; meant to run under ncu (kernel filter by launch index).
import torch, sys, statistics
import vllm  # noqa
from vllm import _custom_ops as ops
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe
from flashinfer.fused_moe.core import ActivationType
dev="cuda"; torch.manual_seed(0)
E,H,I,TOPK=512,2560,640,10
M=int(sys.argv[1]) if len(sys.argv)>1 else 7503; NWARM=int(sys.argv[2]) if len(sys.argv)>2 else 3; NMEAS=int(sys.argv[3]) if len(sys.argv)>3 else 3
w13=torch.randint(0,256,(E,2*I,H//2),device=dev,dtype=torch.uint8); w2=torch.randint(0,256,(E,H,I//2),device=dev,dtype=torch.uint8)
w13_s=(torch.rand(E,2*I,H//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn); w2_s=(torch.rand(E,H,I//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn)
g1=torch.ones(E,device=dev,dtype=torch.float32); g2=torch.ones(E,device=dev,dtype=torch.float32); a1g=torch.ones((),device=dev); a2g=torch.ones((),device=dev)
x=torch.randn(M,H,device=dev,dtype=torch.bfloat16); a_fp4,a_sf=ops.scaled_fp4_quant(x, a1g)
topk_ids=torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32); topk_w=torch.softmax(torch.randn(M,TOPK,device=dev),dim=-1).to(torch.float32)
out=torch.empty(M,H,device=dev,dtype=torch.bfloat16)
def f():
    flashinfer_cutlass_fused_moe(input=a_fp4, token_selected_experts=topk_ids, token_final_scales=topk_w, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
        output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2], input_sf=a_sf, output=out, tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu, use_fused_finalize=True)
for _ in range(NWARM): f()
torch.cuda.synchronize()
s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
for _ in range(NMEAS): f()
e.record(); torch.cuda.synchronize()
fl=2*M*TOPK*(3*H*I); t=s.elapsed_time(e)*1000/NMEAS
print(f"M={M}: {t:.1f} us per layer, {fl/t/1e6:.1f} TFLOPS (measured window, {NMEAS} calls after {NWARM} warmups)", flush=True)
