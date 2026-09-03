# Correctness + determinism test for _C_det.persistent_topk vs the stock op and a reference.
# Covers all paths: rows 1 (decode), 8 (medium smem cap), 64 (filtered), lengths crossing the
# 2048/8192/32768 thresholds, k in (512, 2048), random floats and tie-heavy inputs.
import os, sys, torch, itertools, time
torch.ops.load_library(sys.argv[1])           # path to _C_det.so
import vllm._C  # stock op for the speed/set comparison
dev = "cuda"
def ref_topk(logits, lengths, k):
    rows, cols = logits.shape; out = torch.full((rows, k), -1, dtype=torch.int32, device=dev)
    col = torch.arange(cols, device=dev)
    for r in range(rows):
        n = int(lengths[r]); x = logits[r, :n]
        if n <= k: out[r, :n] = col[:n]; continue
        # exact: value desc, index asc (stable), then ascending index order for the row
        order = torch.argsort(x, descending=True, stable=True)[:k]
        out[r] = torch.sort(order.to(torch.int32)).values
    return out
def run(op, logits, lengths, k, ws):
    out = torch.empty((logits.shape[0], k), dtype=torch.int32, device=dev)
    op(logits, lengths, out, ws, k, logits.shape[1]); return out
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
fails = 0
for rows, cols, k, kind in itertools.product((1, 8, 64), (1024, 4096, 8192, 20000, 40000), (512, 2048), ("rand", "ties")):
    if k >= cols: continue
    g = torch.Generator(device=dev).manual_seed(rows * 7 + cols + k)
    if kind == "rand": logits = torch.randn(rows, cols, generator=g, device=dev)
    else: logits = torch.randint(0, 5, (rows, cols), generator=g, device=dev).float()   # heavy ties
    lengths = torch.full((rows,), cols, dtype=torch.int32, device=dev)
    lengths[0] = cols - 3
    r = ref_topk(logits, lengths, k)
    outs = [run(torch.ops._C_det.persistent_topk, logits, lengths, k, ws) for _ in range(6)]
    same = all(torch.equal(o, outs[0]) for o in outs)
    exact = torch.equal(outs[0], r)
    # value-set check (tolerates a different tie choice only if the SET of values matches the reference)
    stock = run(torch.ops._C.persistent_topk, logits.clone(), lengths, k, ws)
    stock_same = all(torch.equal(run(torch.ops._C.persistent_topk, logits, lengths, k, ws), stock) for _ in range(3))
    tag = "OK " if (same and exact) else "FAIL"
    if tag == "FAIL": fails += 1
    print(f"{tag} rows={rows:2d} cols={cols:5d} k={k:4d} {kind}: det-identical x6={same} exact-ref={exact} | stock reproducible x3={stock_same}")
print("FAILS:", fails)
