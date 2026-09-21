#!/usr/bin/env python3
"""Does the PREALLOCATED (use_cuda_graph=True) b12x path fault where the cached path does not?

vLLM's live path is experts/flashinfer_b12x_moe.py:288 -> B12xMoEWrapper.run(), constructed with
use_cuda_graph=True. Job 300's clean standalone used the default (False), i.e. _WORKSPACE_CACHE.

argv: <max_num_tokens> <run_tokens>   -- separate so the asymmetric case can be tested:
B12xMoEWrapper sizes _static_workspace at min(max_num_tokens*topk, 640) and only allocates
_dynamic_workspace when max_num_tokens*topk > 640. run() then falls back to the 640-row static
workspace whenever the dynamic one is absent, with no capacity check.
Exit 0 clean, 3 CUDA fault, 4 other.
"""
import sys, torch, torch.nn.functional as F
MAXTOK, RUNTOK = int(sys.argv[1]), int(sys.argv[2])
# "neg" reproduces vLLM's profile run, whose routing table is all -1 ("not routed").
# "mask" applies the candidate fix: route invalid slots to expert 0 with weight 0.
MODE = sys.argv[3] if len(sys.argv) > 3 else "real"
E, TOPK, K, N, SF = 512, 10, 2560, 640, 16

def main():
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    torch.manual_seed(7); dev = "cuda"

    w1 = torch.randn(E, 2*N, K, dtype=torch.bfloat16, device=dev)/10
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device=dev)/10
    amax = max(w1.abs().max().item(), w2.abs().max().item())
    s2 = torch.full((E,), amax/(448.0*6.0), device=dev, dtype=torch.float32)
    def q(t, m, k):
        qq, sf = fp4_quantize(t.reshape(E*m, k), global_scale=(1.0/s2[:1]),
                              sf_vec_size=SF, is_sf_swizzled_layout=True)
        return qq.reshape(E, m, k//2), convert_sf_to_mma_layout(sf, m=m, k=k, num_groups=E, sf_vec_size=SF)
    w1q, w1sf = q(w1, 2*N, K); w2q, w2sf = q(w2, K, N)
    del w1, w2; torch.cuda.empty_cache()
    fc2 = torch.tensor([1.0], device=dev, dtype=torch.float32)

    wrap = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=K, intermediate_size=N,
                          use_cuda_graph=True, max_num_tokens=MAXTOK,
                          num_local_experts=E, activation="silu")
    stat = getattr(wrap, "_static_workspace", None); dyn = getattr(wrap, "_dynamic_workspace", None)
    print(f"  max_num_tokens={MAXTOK:5d} -> static={'yes' if stat is not None else 'None'} "
          f"dynamic={'yes' if dyn is not None else 'None'}", flush=True)

    x = torch.randn(RUNTOK, K, dtype=torch.bfloat16, device=dev)
    router = torch.randn(RUNTOK, E, device=dev)
    tw, ids = torch.topk(F.softmax(router, dim=1, dtype=torch.float), TOPK)
    tw = (tw/tw.sum(-1, keepdim=True)).float(); ids = ids.to(torch.int32)
    if MODE in ("neg", "mask"):
        ids = torch.full_like(ids, -1)
        if MODE == "mask":
            valid = ids >= 0
            tw = tw * valid.to(tw.dtype)
            ids = torch.where(valid, ids, torch.zeros_like(ids))
    print(f"  mode={MODE} topk_ids range=[{int(ids.min())},{int(ids.max())}]", flush=True)

    out = wrap.run(x=x, w1_weight=w1q, w1_weight_sf=w1sf, w1_alpha=s2,
                   fc2_input_scale=fc2, w2_weight=w2q, w2_weight_sf=w2sf, w2_alpha=s2,
                   token_selected_experts=ids, token_final_scales=tw)
    torch.cuda.synchronize()
    r = out[0] if isinstance(out, (tuple, list)) else out
    print(f"  run tokens={RUNTOK:5d} rows={RUNTOK*TOPK:6d}  OK  finite={bool(torch.isfinite(r).all())}", flush=True)

try:
    main()
except Exception as e:
    txt = str(e)
    m = f"{type(e).__name__}: {txt.splitlines()[0][:110]}"
    # OOM is a CUDA error but NOT the MMU fault -- conflating them produced a false positive once.
    oom = "out of memory" in txt.lower()
    fault = (not oom) and ("illegal memory access" in txt or "CUDA error" in txt)
    tag = "OOM" if oom else ("FAULT" if fault else "ERROR")
    print(f"  run tokens={RUNTOK:5d} rows={RUNTOK*TOPK:6d}  {tag}  {m}", flush=True)
    sys.exit(3 if fault else (5 if oom else 4))
