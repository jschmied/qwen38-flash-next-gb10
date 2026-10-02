#!/usr/bin/env python3
"""#253's same-bits premise at Flash Next sizes: #212's prompt kernel (windows > 64 rows) and the grouping kernel give
the same fp32 Y for the same rows. D 2560, I 640, 512 experts top 10 + shared, 3-bit and 5-bit trellis, real-like
mixes; rows 65 / 129 / 1000 / 2048. argv: <tensorfold src>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

dev = "cuda"
D, I, E, TOP, CB = 2560, 640, 512, 10, X.CB_MUL1
EA = E + 1
out = {}
for K2 in (6, 10):
    torch.manual_seed(K2)
    t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
    sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
    ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
    for R in (65, 129, 1000, 2048):
        s = X.Scratch(ex, R, TOP + 1)
        g = torch.Generator(device=dev).manual_seed(R)
        x = (torch.randn(R, D, device=dev, generator=g) * 0.5).half()
        routed = torch.rand(R, E, device=dev, generator=g).topk(TOP, dim=1).indices.int()
        pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
        X.PROMPT = "prompt"
        yp = X.routed(x, pk, None, ex, s, None, R).clone()
        X.PROMPT = "group"
        yg = X.routed(x, pk, None, ex, s, None, R).clone()
        X.PROMPT = "prompt"
        eq = bool(torch.equal(yp.view(torch.int32), yg.view(torch.int32)))
        diff = int((yp.view(torch.int32) != yg.view(torch.int32)).sum())
        out[f"K2={K2} R={R}"] = {"equal": eq, "diff_words": diff, "dtype": str(yp.dtype), "finite": bool(torch.isfinite(yp).all())}
        print(json.dumps({f"K2={K2} R={R}": out[f"K2={K2} R={R}"]}), flush=True)
