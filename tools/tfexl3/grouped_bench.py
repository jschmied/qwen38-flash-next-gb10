#!/usr/bin/env python3
"""T13 microbenchmark: TensorFold's EXL3 routed experts (stock grouped kernel) vs a decode-once oracle, Flash Next shapes.
argv: <tensorfold src>. Prints one JSON line per measurement."""
import json, sys, time
sys.path.insert(0, sys.argv[1])
import torch                                                                     # noqa: E402
from tensorfold.cuda.exl3 import experts as X                                    # noqa: E402

torch.manual_seed(0)
dev = "cuda"
D, I, E, TOP, K2, CB = 2560, 640, 512, 10, 6, X.CB_MUL1
EA = E + 1                                    # + the shared expert as expert E (every row's slot 10)
SLOTS = TOP + 1


def trellis(k, n):
    return torch.randint(-32768, 32767, (EA, k // 16, n // 16, 8 * K2), dtype=torch.int16, device=dev)


def sv(n):
    return (torch.randint(0, 2, (EA, n), device=dev).half() * 2 - 1)


ex = X.prepare_stacked(trellis(D, I), trellis(D, I), trellis(I, D), sv(D), sv(D), sv(I), sv(I), sv(I), sv(D), CB)


def picks(R):
    routed = torch.rand(R, E, device=dev).topk(TOP, dim=1).indices.int()
    return torch.cat([routed, torch.full((R, 1), E, dtype=torch.int32, device=dev)], 1).contiguous()


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


ext = X._ext()
for R in (1024, 2048, 4096):
    s = X.Scratch(ex, R, SLOTS)
    x = (torch.randn(R, D, device=dev) * 0.5).half()
    pk = picks(R)
    full = timed(lambda: X.routed(x, pk, None, ex, s, None, R))
    # the two grouped launches alone, with the grouping and rotation done once
    ids, members = s.window(R)
    ext.group(pk, ids, s.count, members, R, SLOTS, EA)
    ext.rot_in(x, x.stride(0), pk, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, SLOTS, EA)
    P = R * SLOTS
    nt, w, sk, pf = s.cfg_gu
    gu = timed(lambda: ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members,
                                   s.z, 2, D, I, P, sk, SLOTS, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1]))
    nt, w, sk, pf = s.cfg_d
    dn = timed(lambda: ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count,
                                   members, s.z, 1, I, D, P, sk, SLOTS, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1]))
    used = int(s.count.item())
    assert ex.k2_gu == (K2, K2) and ex.k2_d == (K2, K2), (ex.k2_gu, ex.k2_d)
    print(json.dumps({"R": R, "routed_ms": round(full, 3), "grouped_gateup_ms": round(gu, 3),
                      "grouped_down_ms": round(dn, 3), "grouped_share": round((gu + dn) / full, 3),
                      "us_per_row_grouped": round(1000 * (gu + dn) / R, 3), "experts_used": used,
                      "cfg_gu": s.cfg_gu, "cfg_d": s.cfg_d}), flush=True)

    # B: decode-once oracle for this window: dequant every used expert's 3 matrices, then padded bmm per expert
    uids = torch.unique(pk.view(-1)).tolist()
    keep = ex.keep                                  # [gate x EA, up x EA, down x EA] trellis tensors
    gate_t, up_t, down_t = keep[:EA], keep[EA:2 * EA], keep[2 * EA:]
    Wg = torch.empty((len(uids), D, I), dtype=torch.float16, device=dev)
    Wu = torch.empty_like(Wg)
    Wd = torch.empty((len(uids), I, D), dtype=torch.float16, device=dev)

    def decode_all():
        for j, e in enumerate(uids):
            ext.dequant(gate_t[e], Wg[j], K2, CB)
            ext.dequant(up_t[e], Wu[j], K2, CB)
            ext.dequant(down_t[e], Wd[j], K2, CB)
    dec = timed(decode_all, reps=3)
    # rows per expert, padded to the routed max (the shared expert alone takes all R rows)
    counts = torch.bincount(pk[:, :TOP].reshape(-1).long(), minlength=E)
    m = int(counts.max())
    xr = torch.randn(len(uids) - 1, m, D, device=dev).half()
    xs = torch.randn(1, R, D, device=dev).half()

    def gemms():
        h = torch.bmm(xr, Wg[:-1]) * torch.bmm(xr, Wu[:-1])
        torch.bmm(h, Wd[:-1])
        hs = torch.bmm(xs, Wg[-1:]) * torch.bmm(xs, Wu[-1:])
        torch.bmm(hs, Wd[-1:])
    mm = timed(gemms)
    print(json.dumps({"R": R, "oracle_decode_ms": round(dec, 3), "oracle_bmm_ms": round(mm, 3),
                      "oracle_total_ms": round(dec + mm, 3), "max_rows_per_expert": m,
                      "speedup_vs_grouped": round((gu + dn) / (dec + mm), 2),
                      "decode_launches": 3 * len(uids)}), flush=True)
    del s, Wg, Wu, Wd
    torch.cuda.empty_cache()
