# One shape per process: mode = a (A only) | u (unfused forward) | f (fused forward) | c (compare both). Usage: test_one.py T varlen mode
import os, sys, torch
import vllm  # noqa
import vllm.third_party.flash_linear_attention.ops.chunk as C
from vllm.third_party.flash_linear_attention.ops.chunk_kkt_solve import chunk_kkt_solve_fwd
from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
T=int(sys.argv[1]); varlen=sys.argv[2]=="1"; mode=sys.argv[3]; dev="cuda"; torch.manual_seed(0); H,HV,K=16,48,128
q=torch.randn(1,T,H,K,device=dev,dtype=torch.bfloat16); k=torch.nn.functional.normalize(torch.randn(1,T,H,K,device=dev).float(),dim=-1).to(torch.bfloat16)
v=torch.randn(1,T,HV,K,device=dev,dtype=torch.bfloat16); beta=torch.rand(1,T,HV,device=dev,dtype=torch.bfloat16); g=(-torch.rand(1,T,HV,device=dev)*0.1).float()
cu=torch.tensor([0,T//3,T],device=dev,dtype=torch.int32) if varlen else torch.tensor([0,T],device=dev,dtype=torch.int32)
OK,OT=C.chunk_scaled_dot_kkt_fwd,C.solve_tril
def kkt_shim(k, beta, g=None, cu_seqlens=None, chunk_indices=None, output_dtype=None, **kw): return chunk_kkt_solve_fwd(k=k,beta=beta,g=g,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,chunk_size=64)
def tril_shim(A, **kw): return A
def fwd(): return C.chunk_gated_delta_rule(q,k,v,g,beta,initial_state=None,output_final_state=True,cu_seqlens=cu,use_qk_l2norm_in_kernel=False)
if mode=="ar":
    gc=chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu); A_ref=OT(A=OK(k=k,beta=beta,g=gc,cu_seqlens=cu,output_dtype=torch.float32),cu_seqlens=cu,output_dtype=k.dtype); torch.cuda.synchronize(); print("vendored A ok", flush=True)
if mode=="an":
    gc=chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu); A_new=chunk_kkt_solve_fwd(k=k,beta=beta,g=gc,cu_seqlens=cu,chunk_size=64); torch.cuda.synchronize(); print("fused A ok", flush=True)
if mode in ("a","c"):
    gc=chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu); A_ref=OT(A=OK(k=k,beta=beta,g=gc,cu_seqlens=cu,output_dtype=torch.float32),cu_seqlens=cu,output_dtype=k.dtype); torch.cuda.synchronize()
    A_new=chunk_kkt_solve_fwd(k=k,beta=beta,g=gc,cu_seqlens=cu,chunk_size=64); torch.cuda.synchronize()
    d=(A_ref.float()-A_new.float()).abs(); print(f"T={T} varlen={varlen} A: max|diff| {d.max().item():.3e} mean {d.mean().item():.2e} |A|max {A_ref.float().abs().max().item():.3f} nan_new={bool(A_new.isnan().any())}", flush=True)
if mode in ("u","c"):
    o1,s1=fwd(); torch.cuda.synchronize(); print(f"unfused forward ok: |o|max {o1.float().abs().max().item():.3f} nan={bool(o1.isnan().any())}", flush=True)
if mode in ("f","c"):
    C.chunk_scaled_dot_kkt_fwd=kkt_shim; C.solve_tril=tril_shim; o2,s2=fwd(); torch.cuda.synchronize(); print(f"fused forward ok: |o|max {o2.float().abs().max().item():.3f} nan={bool(o2.isnan().any())}", flush=True)
    if mode=="c": print(f"forward o max|diff| {(o1.float()-o2.float()).abs().max().item():.3e}  state max|diff| {(s1.float()-s2.float()).abs().max().item():.3e}", flush=True)
