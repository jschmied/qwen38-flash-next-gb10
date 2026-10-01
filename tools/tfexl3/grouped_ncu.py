#!/usr/bin/env python3
"""T13 ncu target: four grouped launches once each (routed top-10, R 1024): stock gate/up, stock down, compacted
gate/up, compacted down. argv: <tensorfold src>."""
import sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB, R = 2560, 640, 512, 10, 6, X.CB_MUL1, 1024
EA = E + 1
t = lambda k, n: torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)  # noqa: E731
sv = lambda n: (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)  # noqa: E731
ex = X.prepare_stacked(t(D, I), t(D, I), t(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
ext = X._ext()
pk = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int().contiguous()
s = X.Scratch(ex, R, TOP)
x = (torch.randn(R, D, device=dev) * 0.5).half()
ids, members = s.window(R)
ext.group(pk, ids, s.count, members, R, TOP, EA)
ext.rot_in(x, x.stride(0), pk, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, TOP, EA)
s.xd.copy_((torch.randn_like(s.xd.float()) * 0.5).half())
torch.cuda.synchronize()
used = int(s.count.item())
m = int((members[:used] >= 0).sum(1).max())
compact = members[:, :-(-m // 16) * 16].contiguous()
P = R * TOP
for mem in (members, compact):
    nt, w, sk, pf = s.cfg_gu
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, mem, s.z, 2, D, I, P, sk, TOP,
                ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
    nt, w, sk, pf = s.cfg_d
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, mem, s.z, 1, I, D, P, sk,
                TOP, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
torch.cuda.synchronize()
print("done", used, m)
