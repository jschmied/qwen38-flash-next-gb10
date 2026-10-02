#!/usr/bin/env python3
"""Every fp16 (F16) matrix an EXL3 Flash Next pack keeps, by shape, and the decode-row bandwidth of _f16 on each shape
(a pool of distinct copies > 120 MB per call series, so nothing is L2-resident). argv: <tensorfold src> <model dir>"""
import collections, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3_mm                          # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

eng = FlashNextEngine(Path(sys.argv[2]), max_len=4096)
w = eng.e.w
seen, shapes = set(), collections.Counter()
sc = None


def walk(o, depth=0):
    global sc
    if depth > 6 or id(o) in seen:
        return
    seen.add(id(o))
    if isinstance(o, exl3_mm.F16):
        shapes[(o.n, o.k, o.sk)] += 1
        sc = o.sc
        return
    if isinstance(o, (list, tuple)):
        for v in o:
            walk(v, depth + 1)
    elif hasattr(o, "__dict__") and not isinstance(o, torch.Tensor):
        for v in vars(o).values():
            walk(v, depth + 1)


walk(w)
total = sum(n * k * 2 * c for (n, k, _), c in shapes.items())
print(json.dumps({"shapes": [{"n": n, "k": k, "sk": s, "count": c, "MB_each": round(n * k * 2 / 2**20, 2)}
                             for (n, k, s), c in shapes.most_common()], "total_MB": round(total / 2**20, 1)}), flush=True)
for (n, k, s), c in shapes.most_common():
    pool = max(2, min(64, (160 << 20) // (n * k * 2) + 1))
    mats = [exl3_mm.F16(torch.randn(n, k, device="cuda").half(), n, k, s, sc) for _ in range(pool)]
    for m in (1, 4, 8):
        x = (torch.randn(m, k, device="cuda")).half()
        out = torch.empty((m, n), dtype=torch.float16, device="cuda")
        for f in mats:
            f(x, out)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        best = 1e9
        for _ in range(3):
            a.record()
            for f in mats:
                f(x, out)
            b.record()
            torch.cuda.synchronize()
            best = min(best, a.elapsed_time(b) / pool)
        print(json.dumps({"n": n, "k": k, "sk": s, "count": c, "M": m, "us": round(best * 1e3, 1),
                          "GBps": round(n * k * 2 / 1e9 / (best / 1e3), 1)}), flush=True)
    del mats
    torch.cuda.empty_cache()
