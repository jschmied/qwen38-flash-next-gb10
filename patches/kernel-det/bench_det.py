# Kernel microbenchmark: stock _C.persistent_topk vs _C_det.persistent_topk.
#   argv[1] = path to _C_det .so.  Rows {1, 64}, n {1k,4k,8k,16k,32k,64k}, k {512, 2048}, random.
import sys, torch, time, itertools
torch.ops.load_library(sys.argv[1]); import vllm  # noqa
dev="cuda"; ws=torch.empty(64*1024*1024, dtype=torch.uint8, device=dev)
import statistics
def bench(op, logits, lengths, k, iters=50, batches=5):
    """5 batches x 50 iterations; returns (median, min, max) of the per-batch mean in us."""
    out=torch.empty((logits.shape[0],k), dtype=torch.int32, device=dev)
    for _ in range(10): op(logits, lengths, out, ws, k, logits.shape[1])
    res=[]
    for _ in range(batches):
        torch.cuda.synchronize(); t0=time.perf_counter()
        for _ in range(iters): op(logits, lengths, out, ws, k, logits.shape[1])
        torch.cuda.synchronize(); res.append((time.perf_counter()-t0)/iters*1e6)
    return statistics.median(res), min(res), max(res)
print(f"{'rows':>4} {'n':>6} {'k':>5} | {'stock med':>9} {'min':>7} {'max':>7} | {'det med':>8} {'min':>7} {'max':>7} | {'ratio(med)':>10} {'ratio(min)':>10}")
cases = [(r, n, k) for r, n, k in itertools.product((1, 64), (1024, 4096, 8192, 16384, 32768, 65536), (512, 2048)) if k < n]
cases += [(r, n, 2048) for r in (2, 4, 8, 16, 24, 32, 48) for n in (8192, 16384, 32768)]
for rows, n, k in cases:
    logits=torch.randn(rows, n, device=dev); lengths=torch.full((rows,), n, dtype=torch.int32, device=dev)
    a=bench(torch.ops._C.persistent_topk, logits, lengths, k); b=bench(torch.ops._C_det.persistent_topk, logits, lengths, k)
    # interleave once more to expose drift: second stock measurement after det
    a2=bench(torch.ops._C.persistent_topk, logits, lengths, k)
    am=min(a[0],a2[0]); amin=min(a[1],a2[1])
    print(f"{rows:4d} {n:6d} {k:5d} | {am:9.1f} {amin:7.1f} {max(a[2],a2[2]):7.1f} | {b[0]:8.1f} {b[1]:7.1f} {b[2]:7.1f} | {b[0]/am:10.2f} {b[1]/amin:10.2f}", flush=True)
