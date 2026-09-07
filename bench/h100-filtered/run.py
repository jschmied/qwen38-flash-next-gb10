# The one question this bundle exists to answer: on a device with >=128 KiB of opt-in shared memory,
# rows > 32 take FilteredTopKRaggedTransform, whose per-row work this PR replaced with the rescanning
# det_select_row. GB10 has 99 KiB and never executes that branch, so it is untested by us.
#
# Prints the device's smem, asserts the Filtered path is actually reachable, then runs correctness
# and cost for both arms over the grid named in the PR body.
import itertools, statistics, time, torch

dev = "cuda"
prop = torch.cuda.get_device_properties(0)
optin = torch.cuda.get_device_properties(0).shared_memory_per_block_optin \
    if hasattr(prop, "shared_memory_per_block_optin") else None
print(f"device: {prop.name}  sm_{prop.major}{prop.minor}  SMs={prop.multi_processor_count}")
print(f"sharedMemPerBlockOptin: {optin} bytes ({(optin or 0)/1024:.0f} KiB)  -> Filtered path "
      f"{'REACHABLE' if (optin or 0) >= 128*1024 else 'NOT reachable -- wrong GPU for this bundle'}")
print(f"torch {torch.__version__}, cuda {torch.version.cuda}\n", flush=True)

ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)
DET = torch.ops._C_det.persistent_topk
BASE = torch.ops._C_base.persistent_topk

def run(op, logits, lengths, k):
    out = torch.empty((logits.shape[0], k), dtype=torch.int32, device=dev)
    op(logits, lengths, out, ws, k, logits.shape[1])
    torch.cuda.synchronize()
    return out

def ref(logits, lengths, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device=dev)
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        o = torch.argsort(logits[r, :n], descending=True, stable=True)[:k]
        out[r, : o.numel()] = torch.sort(o.to(torch.int32)).values
    return out

# --- correctness on the Filtered path (rows > 32 only) -----------------------
print("== CORRECTNESS, rows > 32 " +
      ("(Filtered path)" if (optin or 0) >= 128 * 1024 else "(NOT the Filtered path on this device)") + " ==", flush=True)
print(f"{'case':<38} | {'det self x6':>11} {'exact':>6} | {'base self x3':>12} {'base set':>8}")
fails = 0
for rows, n, k, kind in itertools.product((48, 64), (4096, 8192, 20000, 40000), (512, 2048), ("rand", "ties", "allequal")):
    if k >= n:
        continue
    g = torch.Generator(device=dev).manual_seed(n + k + len(kind))
    if kind == "rand":    logits = torch.randn(rows, n, generator=g, device=dev)
    elif kind == "ties":  logits = torch.randint(0, 5, (rows, n), generator=g, device=dev).float()
    else:                 logits = torch.zeros(rows, n, device=dev)
    lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    r0 = ref(logits, lengths, k)
    d = [run(DET, logits, lengths, k) for _ in range(6)]
    b = [run(BASE, logits, lengths, k) for _ in range(3)]
    dsame = all(torch.equal(x, d[0]) for x in d)
    dexact = torch.equal(d[0], r0)
    bsame = all(torch.equal(x, b[0]) for x in b)
    bset = torch.equal(torch.sort(b[0], dim=1).values, torch.sort(r0, dim=1).values)
    if not (dsame and dexact):
        fails += 1
    print(f"rows={rows:3d} n={n:6d} k={k:5d} {kind:9s} | {str(dsame):>11} {str(dexact):>6} | "
          f"{str(bsame):>12} {str(bset):>8}", flush=True)
print("CORRECTNESS FAILS (det):", fails, flush=True)

# --- cost, the grid named in the PR body ------------------------------------
def bench(op, logits, lengths, k, iters=50, batches=5):
    out = torch.empty((logits.shape[0], k), dtype=torch.int32, device=dev)
    for _ in range(10):
        op(logits, lengths, out, ws, k, logits.shape[1])
    res = []
    for _ in range(batches):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters):
            op(logits, lengths, out, ws, k, logits.shape[1])
        torch.cuda.synchronize(); res.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(res), min(res), max(res)

print("\n== COST (median us of 5 x 50; base measured twice, best taken, to expose drift) ==", flush=True)
print(f"{'rows':>4} {'n':>6} {'k':>5} | {'base med':>8} {'det med':>8} | {'det/base':>8}  (min..max)")
for rows, n, k in [(r, n, k) for r, n, k in
                   itertools.product((64, 128, 256), (4096, 8192, 20000, 40000, 65536),
                                     (512, 1024, 2048)) if k < n]:
    logits = torch.randn(rows, n, device=dev)
    lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    a = bench(BASE, logits, lengths, k)
    d = bench(DET, logits, lengths, k)
    a2 = bench(BASE, logits, lengths, k)
    am = min(a[0], a2[0])
    print(f"{rows:4d} {n:6d} {k:5d} | {am:8.1f} {d[0]:8.1f} | {d[0]/am:8.2f}  "
          f"({d[1]:.1f}..{d[2]:.1f} vs {min(a[1],a2[1]):.1f}..{max(a[2],a2[2]):.1f})", flush=True)

# Rows just below and above the rows>32 switch: the Filtered path turns on here.
print("\n== the rows>32 switch (same n, k) ==", flush=True)
for rows in (16, 32, 33, 48, 64, 96, 128, 256):
    logits = torch.randn(rows, 16384, device=dev)
    lengths = torch.full((rows,), 16384, dtype=torch.int32, device=dev)
    a = bench(BASE, logits, lengths, 2048)
    d = bench(DET, logits, lengths, 2048)
    # The dispatch needs BOTH conditions; on a <128 KiB device no row count reaches
    # the Filtered path, and labelling by row count alone would be a false claim.
    filtered = "yes" if (rows > 32 and (optin or 0) >= 128 * 1024) else "no"
    print(f"rows={rows:3d} n=16384 k=2048 | base {a[0]:7.1f} det {d[0]:7.1f} | {d[0]/a[0]:.2f}x "
          f"| Filtered={filtered}", flush=True)
print("\n== ALL DONE ==", flush=True)
