# Does torch.topk or the MiniMax-M3 MSA triton bitonic top-k meet PR #55122's contract on sm_121,
# and at what cost?  (gau-nernst, https://github.com/vllm-project/vllm/pull/55122#issuecomment-5570130885)
#   argv[1] = path to the branch-built _C_det .so
# Contract under test: exact top-k SET (value desc), emitted in ASCENDING INDEX order, ragged rows,
# int32 out[num_rows, k], bit-identical call to call.
import sys, time, itertools, statistics, torch
torch.ops.load_library(sys.argv[1]); import vllm  # noqa: F401
from vllm.triton_utils import tl, triton
from vllm.models.minimax_m3.common.ops.index_topk import _bitonic_merge
dev = "cuda"
ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=dev)

# --- arm 1: our branch kernel ------------------------------------------------
def run_det(logits, lengths, k, out):
    torch.ops._C_det.persistent_topk(logits, lengths, out, ws, k, logits.shape[1]); return out
def run_stock(logits, lengths, k, out):
    torch.ops._C.persistent_topk(logits, lengths, out, ws, k, logits.shape[1]); return out

# --- arm 2: torch.topk -------------------------------------------------------
# Ragged rows need masking; the contract needs ascending index order, so a sort follows.
NEG = float("-inf")
def run_torch(logits, lengths, k, out):
    n = logits.shape[1]
    mask = torch.arange(n, device=dev)[None, :] < lengths[:, None]
    idx = torch.topk(torch.where(mask, logits, torch.full_like(logits, NEG)), k, dim=1).indices
    out.copy_(torch.sort(idx, dim=1).values.to(torch.int32)); return out
def run_torch_raw(logits, lengths, k, out):           # no mask, no sort: the cheapest possible arm
    torch.topk(logits, k, dim=1); return out

# --- arm 3: the MSA bitonic core, as a plain per-row top-k --------------------
# Same streaming structure as _topk_index_kernel (sort a BLOCK_SIZE_K tile, merge against the
# running winners, keep the top half), with M3's paged/causal/init-local plumbing stripped.
@triton.jit
def _bitonic_topk_row(s_ptr, o_ptr, lengths, n, stride_s_r, stride_o_r,
                      BLOCK_SIZE_K: tl.constexpr, BLOCK_SIZE_T: tl.constexpr):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    row = tl.program_id(0)
    valid = tl.load(lengths + row)
    off_k = tl.arange(0, BLOCK_SIZE_K)
    off_t = tl.arange(0, BLOCK_SIZE_T)
    s_ptrs = s_ptr + row * stride_s_r + off_k
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = off_k < BLOCK_SIZE_K // 2
    n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
    for i in tl.range(0, valid, BLOCK_SIZE_K):
        m = i + off_k < valid
        score = tl.load(s_ptrs, mask=m, other=-1e30).to(tl.float32)
        s_ptrs += BLOCK_SIZE_K
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = tl.where(m, i + off_k + 1, 0), topk_idx
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), j, 2, n_dims)
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims)
            ts = last_topk_score * left_half_mask + topk_score * (1 - left_half_mask)
            ti = last_topk_idx * left_half_mask + topk_idx * (1 - left_half_mask)
            topk_score, topk_idx = _bitonic_merge(ts, ti.to(tl.int32), n_dims, True, n_dims)
        else:
            topk_score, topk_idx = _bitonic_merge(topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims)
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(topk_mask[:, None] * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]), axis=0)
    tl.store(o_ptr + row * stride_o_r + off_t, topk_idx.to(tl.int32))

def bitonic_cfg(k):
    t = triton.next_power_of_2(k)
    return t, t * 2                      # BLOCK_SIZE_K must exceed BLOCK_SIZE_T
def run_bitonic(logits, lengths, k, out):
    T, K = bitonic_cfg(k)
    _bitonic_topk_row[(logits.shape[0],)](logits, out, lengths, logits.shape[1],
                                          logits.stride(0), out.stride(0),
                                          BLOCK_SIZE_K=K, BLOCK_SIZE_T=T,
                                          num_warps=8, num_stages=2)
    return out
def run_bitonic_sorted(logits, lengths, k, out):      # + the ascending-index sort the contract needs
    run_bitonic(logits, lengths, k, out)
    out.copy_(torch.sort(out, dim=1).values); return out

def ref(logits, lengths, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32, device=dev)
    for r in range(logits.shape[0]):
        n = int(lengths[r])
        o = torch.argsort(logits[r, :n], descending=True, stable=True)[:k]
        out[r, :o.numel()] = torch.sort(o.to(torch.int32)).values
    return out

ARMS = [("torch.topk", run_torch), ("bitonic", run_bitonic_sorted), ("det", run_det), ("stock", run_stock)]
print("== CORRECTNESS / DETERMINISM (self x6, and vs the exact reference) ==", flush=True)
hdr = f"{'case':<40}"
for nm, _ in ARMS: hdr += f" | {nm+' self':>14} {'set':>5} {'ord':>4}"
print(hdr, flush=True)
fails = {nm: 0 for nm, _ in ARMS}
cases = [(r, n, k, kind) for r, n, k, kind in itertools.product((1, 8, 64), (1024, 4096, 8192, 20000, 40000), (512, 2048), ("rand", "ties")) if k < n]
cases += [(64, n, 2048, "allequal") for n in (8192, 40000)]
for rows, n, k, kind in cases:
    g = torch.Generator(device=dev).manual_seed(n + k + len(kind))
    if kind == "rand":   logits = torch.randn(rows, n, generator=g, device=dev)
    elif kind == "ties": logits = torch.randint(0, 5, (rows, n), generator=g, device=dev).float()
    else:                logits = torch.zeros(rows, n, device=dev)
    lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    r0 = ref(logits, lengths, k)
    line = f"rows={rows:3d} n={n:6d} k={k:5d} {kind:9s}"
    for nm, fn in ARMS:
        try:
            outs = []
            for _ in range(6):
                o = torch.full((rows, k), -1, dtype=torch.int32, device=dev)
                outs.append(fn(logits, lengths, k, o).clone())
            torch.cuda.synchronize()
        except Exception as e:
            line += f" | {type(e).__name__[:14]:>14} {'-':>5} {'-':>4}"; fails[nm] += 1; continue
        same = all(torch.equal(o, outs[0]) for o in outs)
        sset = torch.equal(torch.sort(outs[0], dim=1).values, torch.sort(r0, dim=1).values)
        sord = torch.equal(outs[0], r0)
        if not (same and sset): fails[nm] += 1
        line += f" | {str(same):>14} {str(sset):>5} {str(sord):>4}"
    print(line, flush=True)
print("FAILS:", fails, flush=True)

print("\n== COST (median us of 5 x 50 launches) ==", flush=True)
def bench(fn, iters=50, batches=5):
    for _ in range(10): fn()
    res = []
    for _ in range(batches):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); res.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(res)
COST = ARMS + [("torch.raw", run_torch_raw), ("bitonic.raw", run_bitonic)]
print(f"{'rows':>4} {'n':>6} {'k':>5} | " + " ".join(f"{nm:>11}" for nm, _ in COST) + " | vs det", flush=True)
for rows, n, k in [(r, n, k) for r, n, k in itertools.product((1, 8, 64), (1024, 4096, 8192, 16384, 32768), (512, 2048)) if k < n]:
    logits = torch.randn(rows, n, device=dev); lengths = torch.full((rows,), n, dtype=torch.int32, device=dev)
    o = torch.empty((rows, k), dtype=torch.int32, device=dev)
    t = {}
    for nm, fn in COST:
        try: t[nm] = bench(lambda fn=fn: fn(logits, lengths, k, o))
        except Exception: t[nm] = None
    cells = " ".join(f"{t[nm]:11.1f}" if t[nm] else f"{'FAIL':>11}" for nm, _ in COST)
    rat = " ".join(f"{nm}={t[nm]/t['det']:.2f}x" for nm, _ in COST if t[nm] and t.get('det'))
    print(f"{rows:4d} {n:6d} {k:5d} | {cells} | {rat}", flush=True)
print("== ALL DONE ==", flush=True)
