# Bench for gau-nernst's blockwise-FP8 ("1d2d": per-row activation scale [M,K/128] + 128x128 weight scale) GEMM kernels
# on GB10, at Flash-Next/finding-67 shapes, against vLLM's CUTLASS sm120 blockwise kernel (single launch and the PR #55180
# 4096-row chunking). Triton kernel copied verbatim from flow-matching@5abc4f1 modelling/linear.py (MIT); CuteDSL kernel
# imported from gn-kernels@ba18197 (path via GN_KERNELS_DIR). Numerics: max |diff| vs the CUTLASS result.
import os, sys, statistics, torch, triton, triton.language as tl
import vllm  # noqa
from vllm import _custom_ops as ops
dev="cuda"; torch.manual_seed(0)
def timeit(fn,n=5,reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)

@triton.jit(do_not_specialize=["M"])
def _fp8_1d2d_kernel(x_ptr, xs_ptr, w_ptr, ws_ptr, o_ptr, M, K: tl.constexpr, stride_xm, stride_xsm, stride_xsk, stride_wn, stride_wsn, stride_wsk, stride_om,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP: tl.constexpr, USE_DOT_SCALED: tl.constexpr):
    tl.static_assert(128 % BLOCK_N == 0); tl.static_assert(128 % BLOCK_K == 0); tl.static_assert(K % BLOCK_K == 0)
    pid_n = tl.program_id(0); pid_m = tl.program_id(1)
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, tl.num_programs(1), tl.num_programs(0), GROUP)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]; offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    x_ptrs = x_ptr + offs_m * stride_xm + offs_k; w_ptrs = w_ptr + offs_n * stride_wn + offs_k[:, None]
    xs_ptrs = xs_ptr + offs_m * stride_xsm; ws_ptrs = ws_ptr + (pid_n * BLOCK_N // 128) * stride_wsn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    if USE_DOT_SCALED:
        sfx = tl.full((BLOCK_M, BLOCK_K // 32), 127, tl.uint8); sfw = tl.full((BLOCK_N, BLOCK_K // 32), 127, tl.uint8)
    for k in range(K // BLOCK_K):
        x = tl.load(x_ptrs, mask_m); w = tl.load(w_ptrs)
        xs = tl.load(xs_ptrs + (k * BLOCK_K // 128) * stride_xsk, mask_m); ws = tl.load(ws_ptrs + (k * BLOCK_K // 128) * stride_wsk)
        if USE_DOT_SCALED: tmp = tl.dot_scaled(x, sfx, "e4m3", w, sfw, "e4m3")
        else: tmp = tl.dot(x, w)
        acc += tmp * (xs.to(tl.float32) * ws.to(tl.float32))
        x_ptrs += BLOCK_K; w_ptrs += BLOCK_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]; offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); mask_m = offs_m < M
    tl.store(o_ptr + offs_m * stride_om + offs_n, acc, mask_m)

def triton_mm(x, xs, w, ws, BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, GROUP=8, dot_scaled=False, warps=4, stages=3):
    M, K = x.shape; N = w.shape[0]; out = x.new_empty(M, N, dtype=torch.bfloat16)
    grid = (N // BLOCK_N, triton.cdiv(M, BLOCK_M), 1)
    _fp8_1d2d_kernel[grid](x, xs, w, ws, out, M, K, x.stride(0), xs.stride(0), xs.stride(1), w.stride(0), ws.stride(0), ws.stride(1), out.stride(0),
                          BLOCK_M, BLOCK_N, BLOCK_K, GROUP, dot_scaled, num_warps=warps, num_stages=stages)
    return out

gn_mm=None
try:
    sys.path.insert(0, os.environ["GN_KERNELS_DIR"])
    from gn_kernels.cutedsl.sm120 import sm120_mm_fp8_1d2d as _gn
    gn_mm=_gn.mm; print("cutedsl kernel imported", flush=True)
except Exception as ex: print(f"cutedsl kernel NOT available: {str(ex).splitlines()[0][:160]}", flush=True)

SHAPES=[("in_proj_qkv",10240,2560),("q_proj",12288,2560),("f67 16384x2560",16384,2560),("f67 5120x5120",5120,5120)]
MS=[int(a) for a in sys.argv[1:]] or [4096,8192,16384]
for tag,Nn,K in SHAPES:
    w=(torch.randn(Nn,K,device=dev)*0.05).to(torch.float8_e4m3fn); ws=(torch.rand(Nn//128,K//128,device=dev)*0.5+0.5).float()
    for M in MS:
        x=(torch.randn(M,K,device=dev)*0.05).to(torch.float8_e4m3fn)
        xs=(torch.rand(K//128,M,device=dev)*0.5+0.5).float().t()   # column-major [M, K/128] (vLLM layout; gn's fp8_quantize makes the same)
        fl=2*M*Nn*K
        ref=ops.cutlass_scaled_mm(x, w.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16)
        t_full=timeit(lambda: ops.cutlass_scaled_mm(x, w.t(), scale_a=xs, scale_b=ws.t(), out_dtype=torch.bfloat16))
        def chunked():
            out=torch.empty(M,Nn,device=dev,dtype=torch.bfloat16)
            for i in range(0,M,4096):
                xs_c=xs[i:i+4096].t().contiguous().t(); out[i:i+4096]=ops.cutlass_scaled_mm(x[i:i+4096], w.t(), scale_a=xs_c, scale_b=ws.t(), out_dtype=torch.bfloat16)
            return out
        t_ch=timeit(chunked)
        print(f"== {tag:14s} N={Nn:5d} K={K:5d} M={M:5d} w={Nn*K/2**20:5.1f}MiB | cutlass full {t_full:8.1f} us {fl/t_full/1e6:6.1f} TF | chunk4096 {t_ch:8.1f} us {fl/t_ch/1e6:6.1f} TF", flush=True)
        for BM,BN,GROUP,warps,stages in ((128,64,8,4,3),(128,128,8,4,3),(128,64,4,4,3),(128,64,16,4,3),(64,64,8,4,3),(128,64,8,8,3),(128,64,8,4,2)):
            try:
                o=triton_mm(x,xs,w,ws,BM,BN,128,GROUP,False,warps,stages); t=timeit(lambda: triton_mm(x,xs,w,ws,BM,BN,128,GROUP,False,warps,stages))
                print(f"   triton BM={BM} BN={BN} G={GROUP:2d} w={warps} s={stages}: {t:8.1f} us {fl/t/1e6:6.1f} TF  x{t_full/t:4.2f} vs full  maxdiff {(o.float()-ref.float()).abs().max():.4f}", flush=True)
            except Exception as ex: print(f"   triton BM={BM} BN={BN} G={GROUP}: ERR {str(ex).splitlines()[0][:100]}", flush=True)
        try:
            o=triton_mm(x,xs,w,ws,128,64,128,8,True,4,3); t=timeit(lambda: triton_mm(x,xs,w,ws,128,64,128,8,True,4,3))
            print(f"   triton dot_scaled BM=128 BN=64 G=8: {t:8.1f} us {fl/t/1e6:6.1f} TF  x{t_full/t:4.2f}  maxdiff {(o.float()-ref.float()).abs().max():.4f}", flush=True)
        except Exception as ex: print(f"   triton dot_scaled: ERR {str(ex).splitlines()[0][:100]}", flush=True)
        if gn_mm is not None:
            try:
                o=gn_mm(x, xs, w, ws); t=timeit(lambda: gn_mm(x, xs, w, ws))
                print(f"   cutedsl sm120 1d2d: {t:8.1f} us {fl/t/1e6:6.1f} TF  x{t_full/t:4.2f}  maxdiff {(o.float()-ref.float()).abs().max():.4f}", flush=True)
            except Exception as ex: print(f"   cutedsl: ERR {str(ex).splitlines()[0][:140]}", flush=True)
