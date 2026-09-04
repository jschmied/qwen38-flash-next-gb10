# Pre-test for the tile-union QSA kernel (plan §5 item 3): does the sparse-attention inner loop get faster with a
# larger M? Strips `_qsa_sparse_paged_gqa_splitk_kernel` to its loop (gather K/V rows by index, QK dot, online
# softmax, PV dot) and runs it with ROWS query rows per program sharing one index list — M = ROWS*16 (12 heads
# padded to 16). Same total work per query row; the only variable is the dot's M. Synthetic data, GB10 shapes:
# head_dim 256, 12 heads per KV head, TOPK 2048 tokens, BLOCK_N 64, page size 16.
import torch, triton, triton.language as tl, statistics, sys
dev="cuda"; torch.manual_seed(0)
H, D, TOPK, BN, PAGE = 16, 256, 2048, 64, 16   # H = padded GQA group
def timeit(fn,n=10,reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)

@triton.jit
def _k(q_ptr, idx_ptr, k_ptr, v_ptr, o_ptr, stride_q_row, stride_idx_row, stride_o_row,
       ROWS: tl.constexpr, H: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, PAGE: tl.constexpr):
    pid = tl.program_id(0)
    M: tl.constexpr = ROWS * H
    m_off = tl.arange(0, M); d_off = tl.arange(0, D); c_off = tl.arange(0, BN)
    row0 = pid * ROWS
    qrow = (row0 + m_off // H) * stride_q_row + (m_off % H) * D
    q = tl.load(q_ptr + qrow[:, None] + d_off[None, :]).to(tl.bfloat16)  # [M, D]
    mx = tl.full((M,), -1.0e20, tl.float32); nrm = tl.zeros((M,), tl.float32); acc = tl.zeros((M, D), tl.float32)
    scale: tl.constexpr = (D ** -0.5) * 1.4426950408889634
    for t in range(0, TOPK, BN):
        cols = t + c_off
        tok = tl.load(idx_ptr + row0 * stride_idx_row + cols)          # shared list for the ROWS rows
        page = tok // PAGE; off = tok % PAGE
        keys = tl.load(k_ptr + (page[None, :] * PAGE + off[None, :]) * D + d_off[:, None])   # [D, BN]
        vals = tl.load(v_ptr + (page[:, None] * PAGE + off[:, None]) * D + d_off[None, :])   # [BN, D]
        s = tl.dot(q, keys) * scale
        nmx = tl.maximum(mx, tl.max(s, 1)); a = tl.math.exp2(mx - nmx); p = tl.math.exp2(s - nmx[:, None])
        acc = tl.dot(p.to(tl.bfloat16), vals, acc=acc * a[:, None]); nrm = nrm * a + tl.sum(p, 1); mx = nmx
    o = acc / nrm[:, None]
    orow = (row0 + m_off // H) * stride_o_row + (m_off % H) * D
    tl.store(o_ptr + orow[:, None] + d_off[None, :], o.to(tl.bfloat16))

def run(N, ROWS, warps, stages=2, BNc=BN):
    q = torch.randn(N, H, D, device=dev, dtype=torch.bfloat16) * 0.1
    ntok = 32768; kc = torch.randn(ntok, D, device=dev, dtype=torch.bfloat16); vc = torch.randn(ntok, D, device=dev, dtype=torch.bfloat16)
    # one index list per program (shared by its ROWS rows), random tokens in the context
    idx = torch.randint(0, ntok, (N, TOPK), device=dev, dtype=torch.int32)
    out = torch.empty(N, H, D, device=dev, dtype=torch.bfloat16)
    def f(): _k[(N // ROWS,)](q, idx, kc, vc, out, q.stride(0), idx.stride(0), out.stride(0), ROWS=ROWS, H=H, D=D, TOPK=TOPK, BN=BNc, PAGE=PAGE, num_warps=warps, num_stages=stages)
    return f
N = 4096
fl = 2 * 2 * N * H * TOPK * D  # QK + PV per query row-group
print(f"N={N} query rows, {H} heads, TOPK={TOPK}, D={D}: {fl/1e9:.0f} GFLOP", flush=True)
# GB10 has 99 KiB shared memory per block: q [M,256] + K [256,BN] + V [BN,256] bf16 x stages must fit.
for ROWS, warps, stages, BNc in ((1,4,2,64),(1,4,1,64),(2,4,1,64),(2,4,2,32),(4,4,1,64),(4,8,1,64),(4,8,1,32),(4,8,2,32),(8,8,1,32),(8,16,1,32),(8,8,1,16)):
    try:
        t = timeit(run(N, ROWS, warps, stages, BNc))
        print(f"  ROWS={ROWS:2d} (M={ROWS*H:3d}) warps={warps:2d} stages={stages} BN={BNc:2d}: {t:9.1f} us  {t/N:6.3f} us/row  {fl/t/1e6:6.1f} TFLOPS", flush=True)
    except Exception as ex:
        print(f"  ROWS={ROWS:2d} (M={ROWS*H:3d}) warps={warps:2d} stages={stages} BN={BNc:2d}: ERR {str(ex).splitlines()[0][:100]}", flush=True)
