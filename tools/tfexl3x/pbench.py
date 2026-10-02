#!/usr/bin/env python3
"""#212's routed() on prompt windows (Flash Next shapes, 3-bit routed + the shared slot at 3 bits, random weights,
4 pick sets rotated): ms per call, output hash, and the prompt kernels' time (torch.profiler). argv: <src> <label>"""
import hashlib, json, sys
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
out = {"label": sys.argv[2]}
for R in (512, 2048):
    s = X.Scratch(ex, R, SLOTS)
    g = torch.Generator(device=dev).manual_seed(R)
    xs = [(torch.randn(R, D, device=dev, generator=g) * 0.5).half() for _ in range(4)]
    pks = []
    for _ in range(4):
        routed = torch.rand(R, E, device=dev, generator=g).topk(TOP, dim=1).indices.int()
        pks.append(torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous())
    h = hashlib.sha256()
    for i in range(4):
        h.update(X.routed(xs[i], pks[i], None, ex, s, None, R).clone().cpu().numpy().tobytes())
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    best = 1e9
    for _ in range(3):
        a.record()
        for _ in range(3):
            for i in range(4):
                X.routed(xs[i], pks[i], None, ex, s, None, R)
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / 12)
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for i in range(4):
            X.routed(xs[i], pks[i], None, ex, s, None, R)
        torch.cuda.synchronize()
    k = {}
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and "prompt" in ev.name:
            n = "gateup" if "prompt_kernel" in ev.name else "down"
            k[n] = k.get(n, 0.0) + ev.device_time_total / 4 / 1e3
    out[f"R{R}"] = {"ms": round(best, 3), "hash": h.hexdigest()[:12], **{n: round(v, 3) for n, v in k.items()}}
print(json.dumps(out), flush=True)
