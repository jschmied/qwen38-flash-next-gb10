import json, torch
out = {}
for M in (1024, 3456):
    a = torch.load(f"/opt/llm/runners/b12xg/y_generic_{M}.pt").float(); b = torch.load(f"/opt/llm/runners/b12xg/y_gated_{M}.pt").float()
    out[str(M)] = {"rel_l2": float((a - b).norm() / a.norm()), "max_abs": float((a - b).abs().max())}
print(json.dumps(out))
