# Fairness follow-up: (1) are the bitonic "set" mismatches a wrong top-k, or just a different tie
# choice?  (2) sweep BLOCK_SIZE_K / num_warps so the bitonic arm is not handicapped by my config.
import os, sys, time, statistics, itertools, torch
torch.ops.load_library(sys.argv[1]); import vllm  # noqa: F401
from vllm.triton_utils import tl, triton
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alt_topk_compare import _bitonic_topk_row, run_det          # reuse the exact kernel under test
dev = "cuda"

def bit(logits, lengths, k, K, warps, stages=2):
    T = triton.next_power_of_2(k)
    out = torch.empty((logits.shape[0], T), dtype=torch.int32, device=dev)
    _bitonic_topk_row[(logits.shape[0],)](logits, out, lengths, logits.shape[1],
                                          logits.stride(0), out.stride(0),
                                          BLOCK_SIZE_K=K, BLOCK_SIZE_T=T,
                                          num_warps=warps, num_stages=stages)
    return out[:, :k]

print("== (1) bitonic set mismatch: wrong values, or a different tie choice? ==", flush=True)
for rows, n, k, kind in [(8, 8192, 512, "ties"), (64, 8192, 2048, "allequal"), (1, 4096, 2048, "ties"), (64, 40000, 512, "ties")]:
    g = torch.Generator(device=dev).manual_seed(n + k + len(kind))
    logits = (torch.zeros(rows, n, device=dev) if kind == "allequal"
              else torch.randint(0, 5, (rows, n), generator=g, device=dev).float())
    lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    o = bit(logits, lengths, k, 2 * triton.next_power_of_2(k), 8)
    vb = torch.sort(torch.gather(logits, 1, o.long()), dim=1, descending=True).values
    vr = torch.sort(logits, dim=1, descending=True).values[:, :k]
    inr = (o >= 0).all().item() and (o < n).all().item()
    dup = all(len(set(r.tolist())) == k for r in o)
    print(f"rows={rows:3d} n={n:6d} k={k:5d} {kind:9s}  values==exact top-k values: {torch.equal(vb, vr)}"
          f"   indices in range: {inr}   no duplicates: {dup}", flush=True)

print("\n== (2) bitonic config sweep (median us of 5 x 50), vs det ==", flush=True)
def bench(fn, iters=50, batches=5):
    for _ in range(10): fn()
    r = []
    for _ in range(batches):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); r.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(r)
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
for rows, n, k in [(1, 8192, 512), (8, 8192, 512), (64, 8192, 512), (8, 32768, 512), (8, 8192, 2048), (8, 32768, 2048)]:
    logits = torch.randn(rows, n, device=dev); lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    od = torch.empty((rows, k), dtype=torch.int32, device=dev)
    d = bench(lambda: torch.ops._C_det.persistent_topk(logits, lengths, od, ws, k, n))
    T = triton.next_power_of_2(k); best = None; cells = []
    for K, w in itertools.product([2 * T, 4 * T], [2, 4, 8, 16]):
        try: t = bench(lambda K=K, w=w: bit(logits, lengths, k, K, w))
        except Exception: cells.append(f"K={K},w={w}:FAIL"); continue
        cells.append(f"K={K},w={w}:{t:.0f}")
        if best is None or t < best[0]: best = (t, K, w)
    b = f"best {best[0]:.1f}us (K={best[1]},warps={best[2]}) = {best[0]/d:.1f}x det" if best else "ALL FAIL"
    print(f"rows={rows:3d} n={n:6d} k={k:5d}  det={d:.1f}us  {b}\n     {'  '.join(cells)}", flush=True)
print("== ALL DONE ==", flush=True)
