#!/usr/bin/env python3
"""NVFP4 grouped experts on a 64-pair-item plan at small row counts: decode kernel (rt 1) vs staged kernel (rt 4), ms
per layer and bit equality. argv: <tensorfold src>"""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda import experts as grouped                                  # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx                                # noqa: E402

dev, E, TOP, D, NI = "cuda", 512, 10, 2560, 640
g = torch.Generator(device=dev).manual_seed(11)


def proj(n, k):
    return (torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g),
            torch.randint(0x28, 0x48, (E, n, k // 16), dtype=torch.uint8, device=dev, generator=g),
            torch.rand(E, device=dev, generator=g) * 0.02 + 0.005)


ex = nvx.make(proj(NI, D), proj(NI, D), proj(D, NI))


def timed(fn, reps=20):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


for rows in (1, 4, 8, 16, 32, 64, 128, 256, 512):
    x = (torch.randn(rows, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    picks = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOP] for _ in range(rows)]).to(torch.int32)
    plan = grouped.Plan(rows, TOP, E, dev, prefill=True)
    grouped.route(picks.contiguous(), plan, 64)
    res, outs = {"rows": rows}, {}
    for rt_rows in (1 << 30, 1):                                   # STAGED_ROWS: never -> rt 1; always -> rt 4
        nvx.STAGED_ROWS = rt_rows
        act = torch.empty((rows * TOP, NI), dtype=torch.bfloat16, device=dev)
        y = torch.empty((rows * TOP, D), dtype=torch.bfloat16, device=dev)
        t = timed(lambda: (nvx.gate_up(x, ex, plan, act, rows), nvx.down(act, ex, plan, y, rows)))
        k = "rt4" if rt_rows == 1 else "rt1"
        res[k + "_ms"] = round(t, 3)
        outs[k] = (act.clone(), y.clone())
    res["equal"] = all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(outs["rt1"], outs["rt4"]))
    print(json.dumps(res), flush=True)
