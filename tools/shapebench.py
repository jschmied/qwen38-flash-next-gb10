"""FP8 scaled_mm vs BF16 F.linear at the shapes decode actually uses.

Guards against the L2-residency lie: a (10240,320) BF16 weight is 6.55 MB and
sits in cache, so a naive loop times the cache, not the kernel. We rotate over
enough distinct weight tensors to exceed cache, and print the roofline time
(N*K*bytes / 273 GB/s) next to every measurement -- anything faster than
roofline is measuring cache and is not a real number.
"""
import torch, time

BW = 273e9
def bench(fn, iters=200, warmup=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t0)/iters*1e6      # us

def run(N, K, label):
    print(f"\n=== {label}  weight ({N}, {K}) ===")
    wb = N*K*2
    ROT = max(8, int(300e6 // wb) + 1)              # rotate past cache
    Wb = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) for _ in range(ROT)]
    Wf, Sc = [], []
    for W in Wb:
        s = (W.float().abs().amax(dim=1)/448.0).clamp(min=1e-12)
        Wf.append((W.float()/s[:,None]).clamp(-448,448).to(torch.float8_e4m3fn))
        Sc.append(s.to(torch.float32))
    print(f"  rotating over {ROT} weights ({ROT*wb/1e6:.0f} MB bf16) to defeat L2")
    print(f"  {'M':>3} {'bf16 us':>9} {'fp8 us':>9} {'speedup':>8} {'bf16 roofline':>14} {'cache-lie?':>11}")
    for M in (1,2,4,8):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        i=[0]
        def f_bf16():
            i[0]=(i[0]+1)%ROT; return torch.nn.functional.linear(x, Wb[i[0]])
        j=[0]
        xf = x.to(torch.float8_e4m3fn)
        sx = torch.ones(M,1,device="cuda",dtype=torch.float32)
        def f_fp8():
            j[0]=(j[0]+1)%ROT
            return torch._scaled_mm(xf, Wf[j[0]].t(), scale_a=sx,
                                    scale_b=Sc[j[0]].view(1,-1), out_dtype=torch.bfloat16)
        tb = bench(f_bf16)
        try: tf = bench(f_fp8)
        except Exception as e: print(f"  {M:>3} {tb:>9.1f}   fp8 FAILED: {type(e).__name__}: {str(e)[:60]}"); continue
        rb = wb/BW*1e6
        lie = "YES" if tb < rb*0.95 else "no"
        print(f"  {M:>3} {tb:>9.1f} {tf:>9.1f} {tb/tf:>7.2f}x {rb:>13.1f} {lie:>11}")

run(10240, 320,  "hyper-connection UP   (what we quantized)")
run(336, 10240,  "hyper-connection DOWN (fused, skinny-GEMM target)")
run(10240, 2560, "GDN in_proj (control: a shape FP8 is known to help)")
