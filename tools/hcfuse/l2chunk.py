"""HC fusion with L2 row sub-chunking: the op's fused path over row sub-chunks of S rows vs one pass, vs today's kernels.
Prints ONE json."""
import os; os.environ["FN_HCFUSE_MIN"] = "1"
import json, torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_silu, _hc_gate_mix
from vllm.models.qwen4_exp.nvidia.ops import hc_fused as hf
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H
torch.manual_seed(3); dev = "cuda"
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16()
def today(res, blk, inj):
    o, xn = _hc_combine_norm(res, blk, inj, w, EPS, HC); d = F.linear(xn, wd)
    return o, _hc_gate_mix(xn, F.linear(_hc_silu(d[:, :R], HC), wu), HC), d
def onepass(res, blk, inj): return hf._combine_mix(res, blk, inj, w, wd, wu, EPS, HC, R)
def chunked(res, blk, inj, S):
    M = res.shape[0]; out = torch.empty_like(res); y = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    d = torch.empty(M, ND, device=dev, dtype=torch.bfloat16)
    for a in range(0, M, S):
        b = min(M, a + S); o_, y_, d_ = hf._combine_mix(res[a:b], blk[a:b], inj[a:b], w, wd, wu, EPS, HC, R)
        out[a:b] = o_; y[a:b] = y_; d[a:b] = d_
    return out, y, d
def chunked_inplace(res, blk, inj, S):  # no copies: the op writes its own outputs; just time the kernels per chunk
    for a in range(0, res.shape[0], S):
        hf._combine_mix(res[a:a+S], blk[a:a+S], inj[a:a+S], w, wd, wu, EPS, HC, R)
def bench(fn, reps=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return round(ts[len(ts) // 2], 4)
out = {}
for M in (3456, 1024):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    r = {"today_ms": bench(lambda: today(res, blk, inj)), "onepass_ms": bench(lambda: onepass(res, blk, inj))}
    o1, y1, d1 = onepass(res, blk, inj)
    for S in (256, 384, 512, 768, 1024):
        if S >= M: continue
        oc, yc, dc = chunked(res, blk, inj, S); oc2, yc2, _ = chunked(res, blk, inj, S)
        r[f"S{S}"] = {"ms_kernels": bench(lambda: chunked_inplace(res, blk, inj, S)),
                      "res_eq": bool(torch.equal(oc, o1)), "y_maxdiff": float((yc.float() - y1.float()).abs().max()),
                      "repro": bool(torch.equal(yc, yc2))}
    out[str(M)] = r
print(json.dumps(out))
