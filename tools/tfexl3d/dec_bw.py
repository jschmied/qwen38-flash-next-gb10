#!/usr/bin/env python3
"""EXL3 routed() at decode/verify window sizes (Flash Next shapes, 3-bit routed trellis, 10 routed + shared): time a
call with a fresh pick set each rep (no L2 reuse between calls), and its effective bandwidth = distinct experts' trellis
bytes / time. Also the grouped kernel alone (torch.profiler). argv: <tensorfold src>"""
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
tg, tu, td = t(D, I), t(D, I), t(I, D)
per_expert = (tg[0].numel() + tu[0].numel() + td[0].numel()) * 2               # trellis bytes an expert
ex = X.prepare_stacked(tg, tu, td, sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
s = X.Scratch(ex, 1024, SLOTS)
POOL = 64
print(json.dumps({"bytes_per_expert_MB": round(per_expert / 2**20, 3)}), flush=True)
for R in (1, 2, 4, 8, 16, 32, 64):
    g = torch.Generator(device=dev).manual_seed(R)
    xs = [(torch.randn(R, D, device=dev, generator=g) * 0.5).half() for _ in range(POOL)]
    pks = []
    for _ in range(POOL):
        routed = torch.rand(R, E, device=dev, generator=g).topk(TOP, dim=1).indices.int()
        pks.append(torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous())
    distinct = sum(int(torch.unique(p).numel()) for p in pks) / POOL            # routed + the shared expert
    for i in range(POOL):
        X.routed(xs[i], pks[i], None, ex, s, None, R)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    best = 1e9
    for _ in range(3):
        a.record()
        for i in range(POOL):
            X.routed(xs[i], pks[i], None, ex, s, None, R)
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / POOL)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for i in range(POOL):
            X.routed(xs[i], pks[i], None, ex, s, None, R)
        torch.cuda.synchronize()
    k = {}
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            name = ev.name.split("(")[0].split("<")[0].replace("void ", "").split("::")[-1]
            k[name] = k.get(name, 0.0) + ev.device_time_total / POOL / 1e3
    grouped = k.get("grouped_kernel", 0.0)
    gb = distinct * per_expert / 1e9
    print(json.dumps({"R": R, "distinct": round(distinct, 1), "routed_ms": round(best, 4),
                      "GBps_routed": round(gb / (best / 1e3), 1), "grouped_ms": round(grouped, 4),
                      "GBps_grouped": round(gb / (grouped / 1e3), 1) if grouped else None,
                      "kernels_ms": {n: round(v, 4) for n, v in sorted(k.items(), key=lambda kv: -kv[1])[:6]}}), flush=True)
