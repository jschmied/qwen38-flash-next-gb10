"""BF16 small-M GEMM microbench (speed-of-light step 3 follow-up). M=4, weights rotated over >= 96 MiB so L2 never
holds them; each candidate timed as a CUDA graph of 48 back-to-back calls; per-call time = graph / 48."""
import json, math, sys, torch, torch.nn.functional as F
import triton, triton.language as tl

SHAPES = {"mixer_down": (324, 10240), "mixer_up": (10240, 320), "shared_gate_up": (1280, 2560),
          "shared_down": (2560, 640), "router": (512, 2560), "in_proj_ba": (96, 2560)}
INMODEL = {"mixer_down": 40.5, "mixer_up": 33.9, "shared_gate_up": 46.0, "shared_down": 23.2, "router": 25.6,
           "in_proj_ba": 17.2}
BW, M, CALLS = 220e9, 4, 48
dev = "cuda"


@triton.jit
def _splitk(x_ptr, w_ptr, o_ptr, N, K, KC: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MR: tl.constexpr):
    pn = tl.program_id(0); pk = tl.program_id(1)
    n = pn * BN + tl.arange(0, BN); m = tl.arange(0, MR)
    acc = tl.zeros((MR, BN), tl.float32)
    for k0 in range(pk * KC, pk * KC + KC, BK):
        k = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + n[:, None] * K + k[None, :], mask=(n[:, None] < N) & (k[None, :] < K), other=0.)
        x = tl.load(x_ptr + m[:, None] * K + k[None, :], mask=k[None, :] < K, other=0.)
        acc += tl.dot(x, tl.trans(w))
    tl.atomic_add(o_ptr + m[:, None] * N + n[None, :], acc, mask=n[None, :] < N)


def triton_fn(bn, bk, split):
    def f(x, w, o32):
        N, K = w.shape
        kc = triton.cdiv(triton.cdiv(K, split), bk) * bk
        o32.zero_()
        _splitk[(triton.cdiv(N, bn), triton.cdiv(K, kc))](x, w, o32, N, K, KC=kc, BN=bn, BK=bk, MR=16)
        return o32.to(torch.bfloat16)
    return f


def bench(fn, x, ws, extra):
    for i in range(3):
        fn(x, ws[i % len(ws)], extra)
    torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for i in range(CALLS):
                fn(x, ws[i % len(ws)], extra)
    torch.cuda.synchronize()
    v = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize(); v.append(a.elapsed_time(b) * 1000 / CALLS)
    v.sort(); del g
    return round(v[len(v) // 2], 1)


def kname(fn, x, w, extra):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(x, w, extra); torch.cuda.synchronize()
    return sorted({e.name[:90] for e in p.events() if e.device_type.name == "CUDA"})


def main():
    import flashinfer
    res = []
    for name, (N, K) in SHAPES.items():
        nbytes = N * K * 2; floor = nbytes / BW * 1e6
        copies = max(CALLS, math.ceil(96 * 2**20 / nbytes))
        ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02 for _ in range(copies)]
        wts = [w.t() for w in ws]                        # (K, N) column-major view for flashinfer
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        x16 = torch.zeros(16, K, device=dev, dtype=torch.bfloat16); x16[:M] = x
        ref = F.linear(x, ws[0]).float()
        cands = {"cublas_default": (lambda x, w, e: F.linear(x, w), x, ws, None)}
        def lt(x, w, e):
            return F.linear(x, w)
        for be in ("cutlass", "cudnn", "cublaslt", "tgv", "tinygemm", "cute-dsl", "auto"):
            cands[f"fi_mm_bf16:{be}"] = ((lambda be: lambda x, w, e: flashinfer.mm_bf16(x, w, backend=be))(be), x, wts, None)
        out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
        cands["fi_tinygemm_bf16"] = (lambda x, w, o: (flashinfer.tinygemm_bf16(x, w, o), o)[1], x, ws, out)
        o32 = torch.empty(16, N, device=dev, dtype=torch.float32)
        for bn in (16, 32, 64):
            for split in (1, 2, 4, 8, 16):
                for bk in (128, 256):
                    if K // split < bk:
                        continue
                    f = triton_fn(bn, bk, split)
                    cands[f"triton_bn{bn}_bk{bk}_s{split}"] = ((lambda f: lambda x, w, o: f(x, w, o)[:M])(f), x16, ws, o32)
        row = {"shape": name, "N": N, "K": K, "floor_us": round(floor, 1), "inmodel_us": INMODEL[name], "cands": {}}
        for cn, (fn, xx, ww, extra) in cands.items():
            try:
                got = fn(xx, ww[0], extra)
                got = got[:M] if got.shape[0] != M else got
                err = (got.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
                if err > 2e-2:
                    row["cands"][cn] = {"error": f"mismatch {err:.3g}"}; continue
                us = bench(fn, xx, ww, extra)
                row["cands"][cn] = {"us": us, "x_floor": round(us / floor, 2)}
            except Exception as ex:
                row["cands"][cn] = {"error": f"{type(ex).__name__}: {str(ex)[:100]}"}
                torch.cuda.synchronize()
        row["cublas_kernels"] = kname(cands["cublas_default"][0], x, ws[0], None)
        ok = {k: v for k, v in row["cands"].items() if "us" in v}
        best = min(ok, key=lambda k: ok[k]["us"])
        row["best"] = best
        print(f"{name:15s} floor {floor:6.1f}  inmodel {INMODEL[name]:5.1f}  cublas {ok['cublas_default']['us']:6.1f}  "
              f"best {best} {ok[best]['us']:.1f} ({ok[best]['x_floor']}x)", flush=True)
        res.append(row)
        del ws, wts; torch.cuda.empty_cache()
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
