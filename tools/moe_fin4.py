# REAL GEMM2 tactic sweep via the autotuner: force one candidate per run (monkeypatched get_valid_tactics) inside autotune(True),
# then measure in normal mode (cache hit). Reports kernel list, GEMM2 epilogue type, finalize-kernel presence, time, checksum.
# Usage: moe_fin4.py M [step]
import torch, sys, collections, re, inspect
import vllm  # noqa
from vllm import _custom_ops as ops
import flashinfer.fused_moe.core as core
from flashinfer.fused_moe.core import ActivationType, cutlass_fused_moe
from flashinfer.autotuner import autotune, AutoTuner
from torch.profiler import profile, ProfilerActivity
dev="cuda"; torch.manual_seed(0); E,H,I,TOPK=512,2560,640,10; M=int(sys.argv[1]); STEP=int(sys.argv[2]) if len(sys.argv)>2 else 4
w13=torch.randint(0,256,(E,2*I,H//2),device=dev,dtype=torch.uint8); w2=torch.randint(0,256,(E,H,I//2),device=dev,dtype=torch.uint8)
w13_s=(torch.rand(E,2*I,H//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn); w2_s=(torch.rand(E,H,I//16,device=dev)*0.5+0.5).to(torch.float8_e4m3fn)
g1=torch.ones(E,device=dev); g2=torch.ones(E,device=dev); a1g=torch.ones((),device=dev); a2g=torch.ones((),device=dev)
x=torch.randn(M,H,device=dev,dtype=torch.bfloat16); a_fp4,a_sf=ops.scaled_fp4_quant(x,a1g)
topk_ids=torch.randint(0,E,(M,TOPK),device=dev,dtype=torch.int32); topk_w=torch.softmax(torch.randn(M,TOPK,device=dev),dim=-1).float(); out=torch.empty(M,H,device=dev,dtype=torch.bfloat16)
def call():
    cutlass_fused_moe(input=a_fp4, token_selected_experts=topk_ids, token_final_scales=topk_w, fc1_expert_weights=w13.view(torch.long), fc2_expert_weights=w2.view(torch.long),
        output_dtype=torch.bfloat16, quant_scales=[a1g, w13_s.view(torch.int32), g1, a2g, w2_s.view(torch.int32), g2], input_sf=a_sf, output=out, tune_max_num_tokens=8192,
        activation_type=ActivationType.Swiglu, use_fused_finalize=True)
call(); torch.cuda.synchronize()
major,minor=torch.cuda.get_device_capability(); mod=core.get_cutlass_fused_moe_module(f"{major*10+minor}")
MR=inspect.getclosurevars(mod.cutlass_fused_moe).nonlocals['MoERunner']; runner=next(iter(MR.runner_dict.values())); n1=runner.get_gemm1_tactic_count(); n2=runner.get_gemm2_tactic_count()
print(f"arch {major*10+minor}: gemm1 tactics {n1}, gemm2 tactics {n2}; occupancy of gemm2 ids: {[runner.get_tactic_occupancy(n1+i) for i in range(0,n2,max(1,n2//8))]}", flush=True)
orig_gvt=MR.get_valid_tactics; FORCE={"t1":None,"t2":None}
def forced_gvt(self, inputs, profile):
    stage=getattr(self,"gemm_idx_for_tuning",None)
    if stage==1 and FORCE["t1"] is not None: return [FORCE["t1"]]
    if stage==2 and FORCE["t2"] is not None: return [FORCE["t2"]]
    return orig_gvt(self, inputs, profile)
MR.get_valid_tactics=forced_gvt
tuner=AutoTuner.get()
def measure(label):
    for _ in range(3): call()
    torch.cuda.synchronize(); N=10
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(N): call()
        torch.cuda.synchronize()
    agg=collections.defaultdict(float); cnt=collections.Counter()
    for ev in p.events():
        if ev.device_type.name=="CUDA": agg[ev.name]+=ev.time_range.elapsed_us()/N; cnt[ev.name]+=1
    tot=sum(agg.values()); gem=sorted([(n,us) for n,us in agg.items() if "GemmUniversal" in n], key=lambda kv:-kv[1]); fin=sum(us for n,us in agg.items() if "finalizeMoeRouting" in n)
    epi=[]
    for n,us in gem:
        toks=sorted(set(re.findall(r"(Scatter|LinearCombination|ScaledAcc[A-Za-z]*)", n))); epi.append(f"{us:.0f}us[{'+'.join(toks)[:40]}]x{cnt[n]//N}")
    cs=out.float().abs().sum().item()
    print(f"  {label}: total {tot:7.0f} us | GEMM {sum(us for _,us in gem):6.0f} us: {' '.join(epi)} | finalize {fin:5.0f} us | act {agg.get(next((n for n in agg if 'doActivation' in n),''),0):.0f} | checksum {cs:.6e}", flush=True)
    return tot
print("=== default (no tuning; fallback tactic -1):"); tuner.clear_cache(); base=measure("default")
print("=== autotuned (full candidate list, tuning mode):"); tuner.clear_cache(); FORCE["t1"]=FORCE["t2"]=None
with autotune(True): call(); torch.cuda.synchronize()
measure("autotuned")
print(f"=== forced GEMM2 tactics (absolute ids {n1}..{n1+n2-1}, step {STEP}); GEMM1 left to the tuner:")
res=[]
for t2 in range(n1, n1+n2, STEP):
    FORCE["t2"]=t2; tuner.clear_cache()
    try:
        with autotune(True): call(); torch.cuda.synchronize()
        res.append((measure(f"gemm2 id {t2:3d} (rel {t2-n1:2d})"), t2))
    except Exception as ex: print(f"  gemm2 id {t2}: ERROR {str(ex).splitlines()[0][:150]}", flush=True)
res.sort(); print("best GEMM2 tactics:", [(t, round(v)) for v,t in res[:5]], "| default total", round(base))
