#!/usr/bin/env python3
"""b12x MoE, one mode per process (argv: generic | gated). gated raises the intermediate-size guard 512 -> 640 BEFORE
any kernel compiles (flashinfer's dynamic-kernel cache key omits the gated decision). Saves outputs for a cross-mode
compare, prints the cached dynamic kernels' classes (witness) and a same-process repeat difference. ONE json."""
import json, sys, torch
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel as mdk
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as mdp
mode = sys.argv[1]
if mode == "gated":
    mdk._GATED_OPTIMIZED_MAX_INTERMEDIATE_SIZE = 640
BUILT = []
CALLS = []
_can = mdk._can_use_gated_optimized_kernel
def _can_rec(**k):
    r = _can(**k); CALLS.append({**{kk: str(v) for kk, v in k.items()}, "result": r}); return r
mdk._can_use_gated_optimized_kernel = _can_rec
for _cls in (mdk.MoEGatedDynamicKernel, mdk._GenericMoEDynamicKernel):
    _orig = _cls.__init__
    def _wrap(self, *a, _o=_orig, _n=_cls.__name__, **k):
        BUILT.append(_n); return _o(self, *a, **k)
    _cls.__init__ = _wrap
from flashinfer.fused_moe import B12xMoEWrapper
from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
E, K, H, I = 512, 10, 2560, 640
dev = "cuda"; torch.manual_seed(0)
w1 = torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8)
s1 = (torch.rand((E, 2 * I, H // 16), device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn)
s2 = (torch.rand((E, H, I // 16), device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn)
sf1 = flashinfer_convert_sf_to_mma_layout(s1.reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
sf2 = flashinfer_convert_sf_to_mma_layout(s2.reshape(E * H, I // 16), m=H, k=I, num_groups=E)
one = torch.ones(E, device=dev, dtype=torch.float32)
wr = B12xMoEWrapper(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, use_cuda_graph=False,
                    max_num_tokens=4096, num_local_experts=E, activation="silu")


def run(x, ids, wts):
    return wr.run(x=x, w1_weight=w1, w1_weight_sf=sf1, w2_weight=w2, w2_weight_sf=sf2, token_selected_experts=ids,
                  token_final_scales=wts, w1_alpha=one, w2_alpha=one, fc2_input_scale=one)


def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


out = {"mode": mode}
for M in (1024, 3456):
    g = torch.Generator(device=dev).manual_seed(M)
    x = torch.randn(M, H, device=dev, generator=g).bfloat16()
    ids = torch.argsort(torch.rand(M, E, device=dev, generator=g), -1)[:, :K].to(torch.int32)
    wts = torch.softmax(torch.randn(M, K, device=dev, generator=g), -1)
    y1 = run(x, ids, wts); y2 = run(x, ids, wts); torch.cuda.synchronize()
    torch.save(y1.cpu(), f"/opt/llm/runners/b12xg/y_{mode}_{M}.pt")
    out[str(M)] = {"ms": bench(lambda: run(x, ids, wts)), "finite": bool(torch.isfinite(y1.float()).all()),
                   "repeat_rel_l2": float((y1.float() - y2.float()).norm() / y1.float().norm())}
out["kernels_built"] = BUILT; out["selector_calls"] = CALLS; out["dyn_cache_len"] = len(mdp._DYNAMIC_KERNEL_CACHE); out["static_cache_len"] = len(mdp._STATIC_KERNEL_CACHE)
print(json.dumps(out))
