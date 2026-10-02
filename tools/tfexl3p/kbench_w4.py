#!/usr/bin/env python3
"""W4A16 only: routed experts of one Flash Next layer (512 x [2560 -> 640 -> 2560], top 10 + the shared slot): the W4A16 grouped
kernel vs the checkpoint-math one, random NVFP4 weights, at decode / verify / prompt row counts. ms per layer (gate|up
+ down; the checkpoint path includes quantizing the rows). argv: <tensorfold src>"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda import experts as grouped                                   # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx                                 # noqa: E402

E, D, NI, TOP = 512, 2560, 640, 10
torch.manual_seed(0)


def proj(n, k):
    return (torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda"),
            torch.randint(0x28, 0x40, (E, n, k // 16), dtype=torch.uint8, device="cuda"),
            torch.rand(E, device="cuda") * 0.01 + 0.002)


gate, up, down = proj(NI, D), proj(NI, D), proj(D, NI)
w4 = nvx.make(gate, up, down)
del gate, up, down
torch.cuda.empty_cache()


def bench(rows, prefill, reps):
    picks = torch.stack([torch.randperm(E)[:TOP] for _ in range(rows)])
    picks = torch.cat([picks, torch.full((rows, 1), E)], 1).to(torch.int32).cuda()
    plan = grouped.Plan(rows, TOP + 1, E + 1, "cuda", prefill=prefill)
    grouped.route(picks, plan, nvx.PREFILL_TILE if prefill else grouped.TILE)
    x = (torch.randn(rows, D, device="cuda") * 0.5).to(torch.bfloat16)
    dt = torch.bfloat16 if prefill else torch.float32
    act = torch.empty((rows * (TOP + 1), NI), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((rows * (TOP + 1), D), dtype=dt, device="cuda")

    def a():
        nvx.gate_up(x, w4, plan, act, rows, skip=E)
        nvx.down(act, w4, plan, y, rows, skip=E)

    out = {}
    for _ in range(3):                                   # alternating rounds
        for name, fn in (("w4a16", a),):
            fn()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            torch.cuda.synchronize()
            out.setdefault(name, []).append(round((time.perf_counter() - t) / reps * 1e3, 4))
    import hashlib
    a()
    torch.cuda.synchronize()
    out["hash"] = hashlib.sha256(y.float().cpu().numpy().tobytes()).hexdigest()[:12]
    return out


for rows, prefill, reps in ((1, False, 200), (4, False, 200), (7, False, 200), (32, False, 100), (2048, True, 10)):
    r = bench(rows, prefill, reps)
    print(json.dumps({"rows": rows, "prefill": prefill, **r}), flush=True)
