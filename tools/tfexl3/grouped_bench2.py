#!/usr/bin/env python3
"""T13 round 2: are empty blocks the cost? grouped launches with members sized to the window (stock) vs compacted to
the real max rows per expert. Flash Next shapes, synthetic trellises (K2 6), R 1024. argv: <tensorfold src>."""
import json, sys
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB, R = 2560, 640, 512, 10, 6, X.CB_MUL1, 1024
EA = E + 1


def trellis(k, n):
    return torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)


def sv(n):
    return (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)


ex = X.prepare_stacked(trellis(D, I), trellis(D, I), trellis(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)
assert ex.k2_gu == (K2, K2) and ex.k2_d == (K2, K2)
ext = X._ext()


def timed(fn, reps=10):
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


def case(name, pk, slots):
    s = X.Scratch(ex, R, slots)
    x = (torch.randn(R, D, device=dev) * 0.5).half()
    ids, members = s.window(R)
    ext.group(pk, ids, s.count, members, R, slots, EA)
    ext.rot_in(x, x.stride(0), pk, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, EA)
    torch.cuda.synchronize()
    used = int(s.count.item())
    rows = (members[:used] >= 0).sum(1)
    m = int(rows.max())
    compact = members[:, :max(16, -(-m // 16) * 16)].contiguous()       # same members, tiles past them dropped
    P = R * slots
    out = {"case": name, "experts": used, "max_rows": m, "mean_rows": round(float(rows.float().mean()), 1),
           "tiles": int(((rows + 15) // 16).sum()), "rows_total": int(rows.sum())}
    zs = {}
    for tag, mem in (("stock", members), ("compact", compact)):
        nt, w, sk, pf = s.cfg_gu
        gu = lambda: ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, mem, s.z, 2,  # noqa: E731
                                 D, I, P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1])
        s.z.zero_()
        gu()
        torch.cuda.synchronize()
        zs[tag + "_gu"] = s.z[:2 * sk * P * I].clone()
        out[tag + "_gateup_ms"] = round(timed(gu), 3)
        nt, w, sk2, pf = s.cfg_d
        dn = lambda: ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, mem,  # noqa: E731
                                 s.z, 1, I, D, P, sk2, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1])
        s.xd.copy_((torch.randn_like(s.xd.float()) * 0.5).half()) if tag == "stock" else None
        s.z.zero_()
        dn()
        torch.cuda.synchronize()
        zs[tag + "_d"] = s.z[:sk2 * P * D].clone()
        out[tag + "_down_ms"] = round(timed(dn), 3)
    import hashlib
    out["hash_gateup"] = hashlib.sha256(zs["compact_gu"].cpu().numpy().tobytes()).hexdigest()[:12]
    out["hash_down"] = hashlib.sha256(zs["compact_d"].cpu().numpy().tobytes()).hexdigest()[:12]
    out["bit_identical_gateup"] = bool(torch.equal(zs["stock_gu"], zs["compact_gu"]))
    out["bit_identical_down"] = bool(torch.equal(zs["stock_d"], zs["compact_d"]))
    out["speedup"] = round((out["stock_gateup_ms"] + out["stock_down_ms"]) /
                           (out["compact_gateup_ms"] + out["compact_down_ms"]), 2)
    c = out["compact_gateup_ms"] + out["compact_down_ms"]
    out["compact_us_per_tile"] = round(1000 * c / out["tiles"], 2)
    out["compact_us_per_row"] = round(1000 * c / out["rows_total"], 3)
    print(json.dumps(out), flush=True)


import os
if os.environ.get("TOPK_SWEEP"):
    for top in (4, 8, 12, 16):
        pk = torch.rand(R, E, device=dev).topk(top, dim=1).indices.int().contiguous()
        case(f"routed{top}", pk, top)
    sys.exit(0)
routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int().contiguous()
case("routed10", routed, TOP)
case("shared_only", torch.full((R, 1), E, dtype=torch.int32, device=dev), 1)
both = torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()
case("routed10+shared", both, TOP + 1)
