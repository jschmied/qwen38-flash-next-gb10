#!/usr/bin/env python3
"""ncu targets (each warmed, then run once): EXL3 dense linear_kernel at decode rows 1 and 4 for Flash Next 5-bit shapes,
the grouped expert kernel at an 8-row window (3-bit), the prefill unpack_kernel (5-bit 2560 x 8192). argv: <src>"""
import sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear                              # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
for K, N in ((2560, 6144), (6144, 2560), (2560, 512)):
    tr = torch.randint(-32768, 32767, (K // 16, N // 16, 8 * 10), dtype=torch.int16)
    lin = Exl3Linear.from_tensors(tr, torch.ones(K).half(), torch.ones(N).half(), "mul1", None, dev)
    for R in (1, 4):
        x = (torch.randn(R, K, device=dev) * 0.5).to(torch.bfloat16)
        out = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
        lin(x, out); lin(x, out)
torch.cuda.synchronize()
D, I, E, TOP, K2, CB, R = 2560, 640, 512, 10, 6, X.CB_MUL1, 8
t = lambda k, n: torch.randint(-32768, 32767, (E + 1, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (E + 1, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
s = X.Scratch(ex, 64, TOP + 1)
x = (torch.randn(R, D, device=dev) * 0.5).half()
pk = torch.cat([torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int(), torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
X.routed(x, pk, None, ex, s, None, R); X.routed(x, pk, None, ex, s, None, R)
torch.cuda.synchronize()
