# Follow-up to kbench (out-of-range: pr faster than base). Two questions per arm and shape:
#  1. which CUDA kernels actually run (torch.profiler names + summed device time per call)
#  2. GPU time without host launch overhead: 20 calls captured in one CUDA graph, replayed
#   argv: <_C_detbase.so> <_C_detpr.so>
import sys, statistics, torch
from torch.profiler import profile, ProfilerActivity
torch.ops.load_library(sys.argv[1]); torch.ops.load_library(sys.argv[2])
import vllm._custom_ops  # noqa: F401
dev = "cuda"; K = 512
WS = torch.empty(1024 * 1024, dtype=torch.uint8, device=dev)
OPS = {"wheel": torch.ops._C.persistent_topk, "base": torch.ops._C_detbase.persistent_topk,
       "pr": torch.ops._C_detpr.persistent_topk}


def exact(x, L, out):
    n = x.shape[1]; col = torch.arange(n, device=dev)
    m = x.masked_fill(col[None, :] >= L.to(torch.int64)[:, None], float("-inf"))
    v, i = torch.topk(m, min(K, n), dim=1)
    out[:, :min(K, n)].copy_(i.masked_fill(torch.isinf(v), -1).to(torch.int32))


def make(rows, n, lk):
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(rows, n, device=dev, generator=g)
    if lk == "full":
        L = torch.full((rows,), n, dtype=torch.int32, device=dev)
    else:
        pos = torch.arange(4 * n - rows, 4 * n, device=dev); L = (pos // 4 + 1).clamp(max=n).to(torch.int32)
    return x, L


def fn(arm):
    if arm == "exact":
        return exact
    op = OPS[arm]
    return lambda x, L, o: op(x, L, o, WS, K, x.shape[1])


print(f"{'shape':22} {'arm':6} | {'graph us/call':>13} | kernels (device us per call)", flush=True)
for rows, n, lk in ((4, 2048, "full"), (4, 8192, "full"), (64, 8192, "full"),
                    (4096, 1024, "causal"), (4096, 2048, "causal"), (4096, 8192, "causal")):
    x, L = make(rows, n, lk); out = torch.empty(rows, K, dtype=torch.int32, device=dev)
    for arm in ("wheel", "base", "pr", "exact"):
        f = fn(arm)
        for _ in range(5): f(x, L, out)
        torch.cuda.synchronize()
        # 1. kernel roster
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            for _ in range(10): f(x, L, out)
            torch.cuda.synchronize()
        ks = {}
        for e in p.events():
            if e.device_type == torch.autograd.DeviceType.CUDA:
                ks[e.name[:60]] = ks.get(e.name[:60], 0.0) + e.device_time / 10
        # 2. graph replay
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph(); gus = float("nan")
        try:
            with torch.cuda.stream(s):
                f(x, L, out)
                with torch.cuda.graph(g, stream=s):
                    for _ in range(20): f(x, L, out)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(5): g.replay()
                b.record(); b.synchronize(); ts.append(a.elapsed_time(b) * 1000 / 100)
            gus = statistics.median(ts)
        except Exception as ex:
            print(f"  graph capture failed for {arm}: {type(ex).__name__}: {str(ex)[:120]}", flush=True)
        kstr = "; ".join(f"{k} {v:.1f}" for k, v in sorted(ks.items(), key=lambda kv: -kv[1])[:4])
        print(f"{rows:4d}x{n:<5d} {lk:6}      {arm:6} | {gus:13.2f} | {kstr}", flush=True)
        del g
print("== KPROBE DONE ==", flush=True)
