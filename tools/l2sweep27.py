# Qwen3.8-27B (prod: vllm-venv-027) GEMM kernels vs M on GB10 (24 MiB L2): per-channel FP8 `cutlass_scaled_mm`
# (attention/GDN projections, 60-120 MiB weights) and FlashInfer CUTLASS NVFP4 `mm_fp4` (MLP, 42.5 MiB) — full-M
# launch vs 4096-row chunks. Timing only (random data); chunk inputs are pre-quantized outside the timed region.
import torch, statistics, sys
import vllm  # noqa
from vllm import _custom_ops as ops
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import pad_nvfp4_weight_for_cutlass, pad_nvfp4_activation_for_cutlass
dev="cuda"; torch.manual_seed(0); CH=4096
MS=[int(a) for a in sys.argv[1:]] or [1024,2048,4096,6144,8192,12288,16384]
def timeit(fn,n=5,reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)
def report(tag,N,K,M,tf,tc):
    fl=2*M*N*K; print(f"  {tag:14s} N={N:5d} K={K:5d} w={N*K/2**20:6.1f}MiB M={M:6d}: full {tf:9.1f} us {fl/tf/1e6:6.1f} TFLOPS | chunk{CH} {tc:9.1f} us {fl/tc/1e6:6.1f} TFLOPS | x{tf/tc:4.2f}", flush=True)
print("== per-channel FP8 (CutlassFP8ScaledMMLinearKernel path)", flush=True)
for tag,N,K in [("in_proj_qkv",10240,5120),("q_proj",12288,5120),("in_proj_z",6144,5120),("out_proj",5120,6144),("fp8 gate_proj",17408,5120)]:
    w=(torch.randn(N,K,device=dev)*0.05).to(torch.float8_e4m3fn); wt=w.t(); ws=torch.rand(1,N,device=dev,dtype=torch.float32)+0.5
    for M in MS:
        a=(torch.randn(M,K,device=dev)*0.05).to(torch.float8_e4m3fn); s=torch.rand(M,1,device=dev,dtype=torch.float32)+0.5
        full=lambda: ops.cutlass_scaled_mm(a,wt,scale_a=s,scale_b=ws,out_dtype=torch.bfloat16)
        def chunked():
            out=torch.empty(M,N,device=dev,dtype=torch.bfloat16)
            for i in range(0,M,CH): out[i:i+CH]=ops.cutlass_scaled_mm(a[i:i+CH],wt,scale_a=s[i:i+CH],scale_b=ws,out_dtype=torch.bfloat16)
            return out
        report(tag,N,K,M,timeit(full),timeit(chunked))
print("== NVFP4 FlashInfer CUTLASS mm_fp4 (FlashInferCutlassNvFp4LinearKernel path)", flush=True)
for tag,N,K in [("gate_up",34816,5120),("down",5120,17408)]:
    w=torch.randn(N,K,device=dev,dtype=torch.bfloat16)*0.05; w_gs=(448*6/w.abs().max().float()).reshape(1).to(dev)
    w_fp4,w_sf=ops.scaled_fp4_quant(w,w_gs,is_sf_swizzled_layout=True); w_fp4,pad=pad_nvfp4_weight_for_cutlass(w_fp4)
    x_gs=torch.tensor([448*6/0.3],device=dev,dtype=torch.float32); alpha=(1.0/(w_gs*x_gs)).to(torch.float32)
    for M in MS:
        x=torch.randn(M,K,device=dev,dtype=torch.bfloat16)*0.05
        def q(t):
            f,sf=ops.scaled_fp4_quant(t,x_gs,is_sf_swizzled_layout=True,backend="flashinfer-cutedsl"); return pad_nvfp4_activation_for_cutlass(f,pad),sf
        xf,xs=q(x); parts=[q(x[i:i+CH]) for i in range(0,M,CH)]
        try:
            full=lambda: flashinfer_scaled_fp4_mm(xf,w_fp4,xs,w_sf,alpha,torch.bfloat16,backend="cutlass")
            def chunked():
                return [flashinfer_scaled_fp4_mm(pf,w_fp4,ps,w_sf,alpha,torch.bfloat16,backend="cutlass") for pf,ps in parts]
            report(tag,N,K,M,timeit(full),timeit(chunked))
        except Exception as ex: print(f"  {tag} M={M}: ERR {str(ex)[:120]}", flush=True)
