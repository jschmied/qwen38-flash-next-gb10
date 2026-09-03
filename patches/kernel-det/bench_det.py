# Kernel microbenchmark: stock _C.persistent_topk vs _C_det.persistent_topk.
#   argv[1] = path to _C_det .so.  Rows {1, 64}, n {1k,4k,8k,16k,32k,64k}, k {512, 2048}, random.
import sys, torch, time, itertools
torch.ops.load_library(sys.argv[1]); import vllm._C  # noqa
dev="cuda"; ws=torch.empty(64*1024*1024, dtype=torch.uint8, device=dev)
def bench(op, logits, lengths, k, iters=50):
    out=torch.empty((logits.shape[0],k), dtype=torch.int32, device=dev)
    for _ in range(5): op(logits, lengths, out, ws, k, logits.shape[1])
    torch.cuda.synchronize(); t0=time.perf_counter()
    for _ in range(iters): op(logits, lengths, out, ws, k, logits.shape[1])
    torch.cuda.synchronize(); return (time.perf_counter()-t0)/iters*1e6
print(f"{'rows':>4} {'n':>6} {'k':>5} {'stock us':>9} {'det us':>8} {'ratio':>6}")
for rows, n, k in itertools.product((1, 64), (1024, 4096, 8192, 16384, 32768, 65536), (512, 2048)):
    if k >= n: continue
    logits=torch.randn(rows, n, device=dev); lengths=torch.full((rows,), n, dtype=torch.int32, device=dev)
    a=bench(torch.ops._C.persistent_topk, logits, lengths, k); b=bench(torch.ops._C_det.persistent_topk, logits, lengths, k)
    print(f"{rows:4d} {n:6d} {k:5d} {a:9.1f} {b:8.1f} {b/a:6.2f}", flush=True)
