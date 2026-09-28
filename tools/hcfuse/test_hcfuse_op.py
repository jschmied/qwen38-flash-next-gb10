"""FNHCFUSE op test: the registered op vs today's kernel sequence. M >= min: fused path, block input identical at
3456 (standalone result) or within today's error; M < min: today's path, must be bit-identical. Prints ONE json."""
import json, torch
import torch.nn.functional as F
from vllm.models.qwen4_exp.nvidia.ops.hc import _hc_combine_norm, _hc_silu, _hc_gate_mix
from vllm.models.qwen4_exp.nvidia.ops.hc_fused import hc_combine_mix_fused, _MIN_M
HC, H, R, EPS, ND = 4, 2560, 320, 1e-6, 336; D = HC * H
torch.manual_seed(1); dev = "cuda"
wd = (torch.randn(ND, D, device=dev) * D ** -0.5).bfloat16(); wu = (torch.randn(D, R, device=dev) * R ** -0.5).bfloat16()
w = (torch.randn(D, device=dev) * 0.1).bfloat16(); out = {"min_m": _MIN_M}
for M in (6, 96, 511, 512, 595, 3456):
    res = torch.randn(M, D, device=dev).bfloat16(); blk = torch.randn(M, H, device=dev).bfloat16()
    inj = torch.randn(M, HC, device=dev).bfloat16()
    o_t, xn = _hc_combine_norm(res, blk, inj, w, EPS, HC); d_t = F.linear(xn, wd)
    y_t = _hc_gate_mix(xn, F.linear(_hc_silu(d_t[:, :R], HC), wu), HC)
    o_f, y_f, d_f = hc_combine_mix_fused(res, blk, inj, w, wd, wu, EPS, HC, R)
    out[str(M)] = {"path": "fused" if M >= _MIN_M else "today", "res_eq": bool(torch.equal(o_f, o_t)),
                   "inj_eq": bool(torch.equal(d_f[:, R:R + HC], d_t[:, R:R + HC])), "y_eq": bool(torch.equal(y_f, y_t)),
                   "y_maxdiff": float((y_f.float() - y_t.float()).abs().max())}
print(json.dumps(out))
