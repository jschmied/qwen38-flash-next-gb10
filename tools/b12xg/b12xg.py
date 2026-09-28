#!/usr/bin/env python3
"""b12x MoE: generic kernel vs the gated-optimized kernel with its intermediate-size guard raised 512 -> 640.
Random NVFP4 weights at Qwen3.8-Flash-Next's MoE shapes; identical inputs to both; relative error and timing.
Prints ONE json."""
import json, torch
import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dynamic_kernel as mdk
from flashinfer.fused_moe import B12xMoEWrapper
from vllm.utils.flashinfer import flashinfer_convert_sf_to_mma_layout
E, K, H, I = 512, 10, 2560, 640
dev = "cuda"; torch.manual_seed(0)


def fp8(shape):
    return (torch.rand(shape, device=dev) * 1.5 + 0.25).to(torch.float8_e4m3fn)


w1 = torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8)
w2 = torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8)
s1 = fp8((E, 2 * I, H // 16)) ; s2 = fp8((E, H, I // 16))
sf1 = flashinfer_convert_sf_to_mma_layout(s1.reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
sf2 = flashinfer_convert_sf_to_mma_layout(s2.reshape(E * H, I // 16), m=H, k=I, num_groups=E)
one = torch.ones(E, device=dev, dtype=torch.float32)


def make(activation="silu"):
    return B12xMoEWrapper(num_experts=E, top_k=K, hidden_size=H, intermediate_size=I, use_cuda_graph=False,
                          max_num_tokens=4096, num_local_experts=E, activation=activation)


def run(wr, x, ids, wts):
    return wr.run(x=x, w1_weight=w1, w1_weight_sf=sf1, w2_weight=w2, w2_weight_sf=sf2, token_selected_experts=ids,
                  token_final_scales=wts, w1_alpha=one, w2_alpha=one, fc2_input_scale=one)


def bench(fn, reps=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


out = {"eligible_before": mdk._can_use_gated_optimized_kernel(activation="silu", sf_vec_size=16, mma_tiler_mn=(128, 128),
                                                                hidden_size=H, intermediate_size=I, num_topk=K)}
cases = {}
for M in (64, 3456):
    x = torch.randn(M, H, device=dev).bfloat16()
    ids = torch.stack([torch.randperm(E, device=dev)[:K] for _ in range(M)]).to(torch.int32)
    wts = torch.softmax(torch.randn(M, K, device=dev), -1)
    cases[M] = (x, ids, wts)
gen = make()  # generic kernel: guard untouched; all generic runs happen before the patch
for M, (x, ids, wts) in cases.items():
    yg = run(gen, x, ids, wts).float()
    out[str(M)] = {"generic_ms": bench(lambda: run(gen, x, ids, wts)), "generic_finite": bool(torch.isfinite(yg).all())}
    cases[M] = (x, ids, wts, yg)
mdk._GATED_OPTIMIZED_MAX_INTERMEDIATE_SIZE = 640
out["eligible_after"] = mdk._can_use_gated_optimized_kernel(activation="silu", sf_vec_size=16, mma_tiler_mn=(128, 128),
                                                              hidden_size=H, intermediate_size=I, num_topk=K)
try:
    gat = make()
except Exception as e:
    gat = None; out["gated_build_error"] = f"{type(e).__name__}: {str(e)[:300]}"
for M, (x, ids, wts, yg) in cases.items():
    r = out[str(M)]
    if gat is None: continue
    try:
        yf = run(gat, x, ids, wts).float(); torch.cuda.synchronize()
        r["gated_ms"] = bench(lambda: run(gat, x, ids, wts))
        r["rel_l2"] = float((yf - yg).norm() / yg.norm())
        r["max_abs"] = float((yf - yg).abs().max()); r["finite"] = bool(torch.isfinite(yf).all())
    except Exception as e:
        r["gated_run_error"] = f"{type(e).__name__}: {str(e)[:300]}"
out["kernel_types"] = [type(getattr(w, "_kernel", None)).__name__ for w in (gen, gat) if w is not None]
print(json.dumps(out))
