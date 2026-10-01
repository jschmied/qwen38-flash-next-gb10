#!/usr/bin/env python3
"""routed() at decode-sized windows (R 1/2/4/8/16), Flash Next shapes: 5 measurements of 2,000 calls each per R.
argv: <tensorfold src> <label>. Prints one JSON line per R (median and min µs a call)."""
import json, statistics, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB = 2560, 640, 512, 10, 6, X.CB_MUL1
EA, SLOTS = E + 1, TOP + 1
t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
s = X.Scratch(ex, 16, SLOTS)
for R in (1, 2, 4, 8, 16):
    x = (torch.randn(R, D, device=dev) * 0.5).half()
    routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int()
    pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
    for _ in range(200):
        X.routed(x, pk, None, ex, s, None, R)
    us = []
    for _ in range(5):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(2000):
            X.routed(x, pk, None, ex, s, None, R)
        b.record()
        torch.cuda.synchronize()
        us.append(1000 * a.elapsed_time(b) / 2000)
    print(json.dumps({"label": sys.argv[2], "R": R, "median_us": round(statistics.median(us), 2),
                      "min_us": round(min(us), 2)}), flush=True)
