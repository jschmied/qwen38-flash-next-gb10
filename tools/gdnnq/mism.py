"""Dump the rare A mismatches between fused and today at T=3456: the bf16 y (from rmsnorm_fn), both fp8 codes, the scale."""
import json, sys, torch
sys.path.insert(0, "/opt/llm/runners/gdnnq")
import importlib.util
spec = importlib.util.spec_from_file_location("g", "/opt/llm/runners/gdnnq/gdnnq_lib.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
torch.manual_seed(0); dev = "cuda"; NH, D = 48, 128
out = []
for wdt in (torch.float32, torch.bfloat16):
    w = (1.0 + 0.1 * torch.randn(D, device=dev)).to(wdt)
    x = torch.randn(3456, NH, D, device=dev).bfloat16() * 3; z = torch.randn(3456, NH, D, device=dev).bfloat16()
    y = rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu").flatten(-2)
    qa, sa = g.today(x, z, w, 1e-6); qb, sb = g.fused(x, z, w, 1e-6)
    idx = (qa.view(torch.uint8) != qb.view(torch.uint8)).nonzero()[:8]
    for t, c in idx.tolist():
        h = c // 128
        out.append({"w": str(wdt)[6:], "tok": t, "col": c, "y": float(y[t, c]), "q_today": float(qa[t, c].float()), "q_fused": float(qb[t, c].float()),
                    "s_today": float(sa[t, h]), "s_fused": float(sb[t, h]), "y_over_s": float(y[t, c].float() / sa[t, h])})
print(json.dumps(out))
