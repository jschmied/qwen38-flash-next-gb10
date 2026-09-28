"""Crossover sweep: fused op (FN_HCFUSE_MIN=1, set before import) vs today's kernel sequence. Prints ONE json."""
import os; os.environ["FN_HCFUSE_MIN"] = "1"
import json, torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_silu, _hc_gate_mix
from vllm.models.qwen4_exp.nvidia.ops.hc_fused import hc_combine_mix_fused
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H
torch.manual_seed(2); dev = "cuda"
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()


def today(res, blk, inj):
    o, xn = _hc_combine_norm(res, blk, inj, w, EPS, HC); d = F.linear(xn, wd)
    return o, _hc_gate_mix(xn, F.linear(_hc_silu(d[:, :R], HC), wu), HC), d


def bench(fn, reps=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)


out = {}
for M in (512, 576, 608, 640, 672, 704, 768, 1024, 1536):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    t = bench(lambda: today(res, blk, inj)); f = bench(lambda: hc_combine_mix_fused(res, blk, inj, w, wd, wu, EPS, HC, R))
    out[M] = {"today_ms": t, "fused_ms": f, "delta_pct": round(100 * (f / t - 1), 1)}
print(json.dumps(out))
