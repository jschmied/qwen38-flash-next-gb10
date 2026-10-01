#!/usr/bin/env python3
"""Kernel breakdown of TensorFold's EXL3 routed() (Flash Next shapes) via torch.profiler. argv: <tensorfold src>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from torch.profiler import ProfilerActivity, profile                             # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB = 2560, 640, 512, 10, 6, X.CB_MUL1
EA, SLOTS = E + 1, TOP + 1
t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
for R in (1024, 2048):
    s = X.Scratch(ex, R, SLOTS)
    x = (torch.randn(R, D, device=dev) * 0.5).half()
    routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int()
    pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
    for _ in range(3):
        X.routed(x, pk, None, ex, s, None, R)
    torch.cuda.synchronize()
    reps = 5
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            X.routed(x, pk, None, ex, s, None, R)
        torch.cuda.synchronize()
    rows = {}
    for e in prof.key_averages():
        if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0):
            us = getattr(e, "self_device_time_total", 0) or getattr(e, "self_cuda_time_total", 0)
            if us:
                rows[e.key[:60]] = rows.get(e.key[:60], 0) + us / reps
    total = sum(rows.values())
    print(json.dumps({"R": R, "total_ms": round(total / 1000, 3), "per_row_us": round(total / R, 2),
                      "kernels_ms": {k: round(v / 1000, 3) for k, v in sorted(rows.items(), key=lambda kv: -kv[1])}}),
          flush=True)
    del s
    torch.cuda.empty_cache()
