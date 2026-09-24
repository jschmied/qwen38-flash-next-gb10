import sys, torch, torch.nn.functional as F
sys.path.insert(0, "/opt/llm/runners/bf16mb")
import fn_bf16sk_venv as m
bad = 0
for (N, K) in m.CONFIGS:
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
    for M in list(range(1, 18)):
        big = torch.randn(M, K + 4, device="cuda", dtype=torch.bfloat16)
        for x in (big[:, :K].contiguous(), big[:, :K]):           # contiguous and strided (lora-slice-like)
            ref = F.linear(x, w).float(); got = m._fn_bf16sk_impl(x, w).float()
            err = ((got - ref).abs().max() / ref.abs().max()).item()
            rep = len({m._fn_bf16sk_impl(x, w).view(torch.int16).sum().item() for _ in range(5)})
            if err > 1e-2 or rep != 1 or got.shape != ref.shape:
                bad += 1; print("BAD", N, K, M, x.is_contiguous(), err, rep)
print("cases bad:", bad)
