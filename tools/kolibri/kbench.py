"""One layer's prompt experts on random FP8 weights of Kolibri's shape, routed like a 2,048-row chunk: kernel timing."""

import sys
import time

import torch

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.fp8 import experts as fp8x

rows, e, width, dims, k = int(sys.argv[1]) if len(sys.argv) > 1 else 2048, 385, 512, 2560, 7
g = torch.Generator().manual_seed(0)


def fp8(n, kk):
    return ((torch.randn(e, n, kk, generator=g) * 2).clamp(-400, 400).to(torch.float8_e4m3fn).cuda(),
            (torch.rand(e, n // 128, kk // 128, generator=g) * 0.01).cuda())


ex = fp8x.make(fp8(width, dims), fp8(width, dims), fp8(dims, width))
x = (torch.randn(rows, dims, generator=g) * 0.5).to(torch.bfloat16).cuda()
picks = torch.cat([torch.stack([torch.randperm(384, generator=g)[:6] for _ in range(rows)]),
                   torch.full((rows, 1), 384)], 1).to(torch.int32).cuda()
plan = grouped.Plan(rows, k, e, "cuda", prefill=True)
grouped.route(picks, plan, fp8x.PROMPT_TILE)
act = torch.empty((rows * k, width), dtype=torch.bfloat16, device="cuda")
y = torch.empty((rows * k, dims), dtype=torch.bfloat16, device="cuda")
for _ in range(3):
    fp8x.gate_up(x, ex, plan, act, rows)
    fp8x.down(act, ex, plan, y, rows)
torch.cuda.synchronize()
for name, fn in (("gate_up", lambda: fp8x.gate_up(x, ex, plan, act, rows)), ("down", lambda: fp8x.down(act, ex, plan, y, rows))):
    t0 = time.perf_counter()
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 100
    flop = rows * k * 2 * dims * width * (2 if name == "gate_up" else 1)
    print(f"{name}: {ms:.3f} ms, {flop / ms / 1e9:.1f} TFLOPS")
