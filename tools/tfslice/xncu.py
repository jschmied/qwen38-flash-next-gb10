#!/usr/bin/env python3
"""One gate|up and one down launch of the NVFP4 grouped experts (8,192 rows, Flash Next routing, tile from argv) for ncu.
argv: <tensorfold src> <tile>"""
import sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda import experts as grouped                                  # noqa: E402
from tensorfold.cuda.nvfp4 import experts as nvx                                # noqa: E402

dev, E, TOP, D, NI, rows, tile = "cuda", 512, 10, 2560, 640, 8192, int(sys.argv[2])
g = torch.Generator(device=dev).manual_seed(11)


def proj(n, k):
    return (torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device=dev, generator=g),
            torch.randint(0x28, 0x48, (E, n, k // 16), dtype=torch.uint8, device=dev, generator=g),
            torch.rand(E, device=dev, generator=g) * 0.02 + 0.005)


ex = nvx.make(proj(NI, D), proj(NI, D), proj(D, NI))
x = (torch.randn(rows, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
picks = torch.stack([torch.randperm(E, device=dev, generator=g)[:TOP] for _ in range(rows)]).to(torch.int32)
plan = grouped.Plan(rows, TOP, E, dev, prefill=True)
grouped.route(picks.contiguous(), plan, tile)
act = torch.empty((rows * TOP, NI), dtype=torch.bfloat16, device=dev)
y = torch.empty((rows * TOP, D), dtype=torch.bfloat16, device=dev)
for _ in range(2):
    nvx.gate_up(x, ex, plan, act, rows)
    nvx.down(act, ex, plan, y, rows)
torch.cuda.synchronize()
