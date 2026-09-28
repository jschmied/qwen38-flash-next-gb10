import json, torch, importlib.util
spec = importlib.util.spec_from_file_location("g", "/opt/llm/runners/gdnnq/gdnnq_dbg.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import rmsnorm_fn
torch.manual_seed(0); dev = "cuda"; NH, D = 48, 128; out = {}
for wdt in (torch.float32, torch.bfloat16):
    w = (1.0 + 0.1 * torch.randn(D, device=dev)).to(wdt)
    x = torch.randn(3456, NH, D, device=dev).bfloat16() * 3; z = torch.randn(3456, NH, D, device=dev).bfloat16()
    ya = rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None, norm_before_gate=True, activation="silu").reshape(-1, D)
    qa, sa = g.today(x, z, w, 1e-6)
    for fpf in (True, False):
        qb, sb = g.fused(x, z, w, 1e-6, fpf=fpf); yb = g.fused.ydbg
        out[f"{str(wdt)[6:]}-fpfusion{fpf}"] = {"y_diff": int((ya != yb).sum()), "A_mismatch": int((qa.view(torch.uint8) != qb.view(torch.uint8)).sum()), "scale_eq": bool(torch.equal(sa, sb))}
print(json.dumps(out))
