#!/usr/bin/env python3
"""ncu target: routed() at R 1024 with two member tiles a program (MS 2), then forced one (MS 1). argv: <src>"""
import sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB, R = 2560, 640, 512, 10, 6, X.CB_MUL1, 1024
EA, SLOTS = E + 1, TOP + 1
t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
s = X.Scratch(ex, R, SLOTS)
x = (torch.randn(R, D, device=dev) * 0.5).half()
routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int()
pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
X.routed(x, pk, None, ex, s, None, R)                                            # MS 2 (2 grouped launches)
X.SUBTILES_FROM = 1 << 30
X.routed(x, pk, None, ex, s, None, R)                                            # MS 1
torch.cuda.synchronize()
print("done")
