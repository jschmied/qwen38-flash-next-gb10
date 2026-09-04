# GB10 L2 hypothesis for the NVFP4 grouped MoE GEMM (Flash-Next geometry: 512 experts, top-10, 2560x640):
# rows per expert = M*TOPK/E; once that crosses one M-tile the expert's weight is streamed once more from DRAM.
# Sweep M around the 128-row boundary with balanced (exact rows/expert) and random routing; autotuner default path.
import torch, statistics, sys, math
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
wbytes=w13.numel()+w2.numel()+w13_s.numel()+w2_s.numel()
print(f"expert weight per layer: {wbytes/1e9:.2f} GB -> single-stream floor at 273 GB/s = {wbytes/273e9*1e6:.0f} us", flush=True)
def make_ids(M, mode):
    if mode=="random": return torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32)
    return (torch.arange(M*TOPK,device=dev)%E).view(M,TOPK).to(torch.int32)   # balanced: exactly M*TOPK/E rows per expert (+-1)
def run(M, mode):
    x=torch.randn(M,H,device=dev,dtype=torch.bfloat16); a_fp4,a_sf=ops.scaled_fp4_quant(x, a1g)
    topk_ids=make_ids(M,mode); topk_w=torch.softmax(torch.randn(M,TOPK,device=dev),dim=-1).to(torch.float32)
    out=torch.empty(M,H,device=dev,dtype=torch.bfloat16)
    def f():
        flashinfer_cutlass_fused_moe(input=a_fp4, token_selected_experts=topk_ids, token_final_scales=topk_w, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
            output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2], input_sf=a_sf, output=out, tune_max_num_tokens=max(8192,M),
            activation_type=ActivationType.Swiglu, use_fused_finalize=True)
    return f
MS=[int(a) for a in sys.argv[1:]] or [2048,3277,4096,6144,6554,6656,7503,8192,9830,12288,13107,13312,16384]
for mode in ("balanced","random"):
    print(f"== routing: {mode}", flush=True)
    for M in MS:
        r=M*TOPK/E; fl=2*M*TOPK*(3*H*I); t=timeit(run(M,mode))
        print(f"  M={M:6d} rows/expert={r:6.1f} tiles@128={math.ceil(r/128)} tiles@64={math.ceil(r/64)}: {t:9.1f} us  {fl/t/1e6:6.1f} TFLOPS  {t/M:6.3f} us/token  x_floor={t/(wbytes/273e9*1e6):4.2f}", flush=True)
