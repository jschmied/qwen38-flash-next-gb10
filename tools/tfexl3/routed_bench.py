#!/usr/bin/env python3
"""T13: time TensorFold's EXL3 routed() on Flash Next shapes (10 routed + the shared expert) and hash its output.
argv: <tensorfold src>. Same script on stock and branch: routed()'s signature is unchanged."""
import hashlib, json, sys
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
assert ex.k2_gu == (K2, K2)
s = X.Scratch(ex, 1024, SLOTS)


def timed(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


for R in (1, 8, 64, 1024):
    g = torch.Generator(device=dev).manual_seed(R)
    x = (torch.randn(R, D, device=dev, generator=g) * 0.5).half()
    routed = torch.rand(R, E, device=dev, generator=g).topk(TOP, dim=1).indices.int()
    pk = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
    y = X.routed(x, pk, None, ex, s, None, R).clone()
    ms = timed(lambda: X.routed(x, pk, None, ex, s, None, R), 20 if R < 1024 else 10)
    print(json.dumps({"R": R, "routed_ms": round(ms, 4), "hash": hashlib.sha256(y.cpu().numpy().tobytes()).hexdigest()[:16],
                      "finite": bool(torch.isfinite(y).all())}), flush=True)
