#!/usr/bin/env python3
"""ncu target: #212's prompt_kernel (routed, 2,048 rows, 3-bit trellis) and the dense prefill unpack_kernel (5-bit,
2560 x 8192), each warmed once then run once. argv: <tensorfold src>"""
import sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402
from tensorfold.cuda.exl3.linear import Exl3Linear                              # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB, R = 2560, 640, 512, 10, 6, X.CB_MUL1, 2048
EA = E + 1
t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
s = X.Scratch(ex, R, TOP + 1)
x = (torch.randn(R, D, device=dev) * 0.5).half()
routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int()
pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
for _ in range(2):
    X.routed(x, pk, None, ex, s, None, R)
torch.cuda.synchronize()
K, N, BITS = 2560, 8192, 5
tr = torch.randint(-32768, 32767, (K // 16, N // 16, 16 * BITS), dtype=torch.int16)
lin = Exl3Linear.from_tensors(tr, torch.ones(K).half(), torch.ones(N).half(), "mul1")
w = torch.empty((K, N), dtype=torch.float16, device=dev)
for _ in range(2):
    lin.unpack(w)
torch.cuda.synchronize()
