# Hyper-connection Triton kernels on GB10: stock `_hc_combine_norm` / `_hc_gate_mix` vs a torch bandwidth floor
# and re-tiled variants (one program per row handling all HC streams, block output read once). Numerics vs stock.
import torch, triton, triton.language as tl, statistics, sys
import vllm  # noqa
from vllm.models.qwen3_8_flash_next.nvidia.ops import hc as HC
dev="cuda"; torch.manual_seed(0); HCN, HD = 4, 2560; DIM=HCN*HD; EPS=1e-6
def timeit(fn,n=10,reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)

@triton.jit
def _combine_norm_v2(block_ptr, res_ptr, inj_ptr, w_ptr, out_ptr, y_ptr, stride_block, stride_res, stride_inj, stride_out, stride_y,
                     HC_DIM: tl.constexpr, HC: tl.constexpr, W_SHARED: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    # one program per row; all HC streams; block read once per tile; two passes over tiles (sum_sq, then y) re-reading `out` from L2.
    row = tl.program_id(0)
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    offs_hc = tl.arange(0, HC_PAD); mask_hc = offs_hc < HC
    inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)              # [HC_PAD]
    sum_sq = tl.zeros([HC_PAD], dtype=tl.float32)
    for t in range(0, HC_DIM, BLOCK):
        offs_inner = t + tl.arange(0, BLOCK); m = offs_inner < HC_DIM
        blk = tl.load(block_ptr + row * stride_block + offs_inner, m, other=0.0).to(tl.float32)   # [BLOCK]
        offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]; m2 = mask_hc[:, None] & m[None, :]
        res = tl.load(res_ptr + row * stride_res + offs, m2, other=0.0).to(tl.float32)             # [HC_PAD, BLOCK]
        out = (res + blk[None, :] * inj[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row * stride_out + offs, out, mask=m2)
        o32 = out.to(tl.float32); sum_sq += tl.sum(o32 * o32, axis=1)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)                                                            # [HC_PAD]
    for t in range(0, HC_DIM, BLOCK):
        offs_inner = t + tl.arange(0, BLOCK); m = offs_inner < HC_DIM
        offs = offs_hc[:, None] * HC_DIM + offs_inner[None, :]; m2 = mask_hc[:, None] & m[None, :]
        out = tl.load(out_ptr + row * stride_out + offs, m2, other=0.0).to(tl.float32)
        if W_SHARED:
            w = tl.load(w_ptr + offs_inner, m, other=0.0).to(tl.float32)[None, :]
        else:
            w = tl.load(w_ptr + offs, m2, other=0.0).to(tl.float32)
        y = out * rrms[:, None]; y += y * w
        tl.store(y_ptr + row * stride_y + offs, y, m2)

def combine_norm_v2(residual, block_output, inj, w, eps, hc, BLOCK=512, num_warps=4):
    N, D = residual.shape; hd = D // hc; out = residual.new_empty(residual.shape); y = residual.new_empty(residual.shape)
    _combine_norm_v2[(N,)](block_output, residual, inj, w, out, y, block_output.stride(0), residual.stride(0), inj.stride(0), out.stride(0), y.stride(0),
                           HC_DIM=hd, HC=hc, W_SHARED=w.numel()==hd, EPS=eps, BLOCK=BLOCK, num_warps=num_warps)
    return out, y

@triton.jit
def _gate_mix_v2(x_ptr, g_ptr, y_ptr, stride_x, stride_g, stride_y, HC_DIM: tl.constexpr, HC: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    # ROWS rows x BLOCK columns per program, HC streams summed in registers.
    pid_r = tl.program_id(0); pid_c = tl.program_id(1)
    rows = pid_r * ROWS + tl.arange(0, ROWS); cols = pid_c * BLOCK + tl.arange(0, BLOCK); mc = cols < HC_DIM
    acc = tl.zeros([ROWS, BLOCK], dtype=tl.float32)
    for s in tl.static_range(HC):
        offs = rows[:, None] * stride_x + s * HC_DIM + cols[None, :]
        g = tl.load(g_ptr + rows[:, None] * stride_g + s * HC_DIM + cols[None, :], mc[None, :], other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mc[None, :], other=0.0).to(tl.float32)
        acc += tl.sigmoid(g) * x
    acc /= HC
    tl.store(y_ptr + rows[:, None] * stride_y + cols[None, :], acc, mc[None, :])

def gate_mix_v2(x, gate, hc, ROWS=4, BLOCK=512, num_warps=4):
    N, D = gate.shape; hd = D // hc; assert N % ROWS == 0; out = x.new_empty(N, hd)
    _gate_mix_v2[(N // ROWS, triton.cdiv(hd, BLOCK))](x, gate, out, x.stride(0), gate.stride(0), out.stride(0), HC_DIM=hd, HC=hc, ROWS=ROWS, BLOCK=BLOCK, num_warps=num_warps)
    return out

for N in [int(a) for a in sys.argv[1:]] or (4096, 8192):
    res = torch.randn(N, DIM, device=dev, dtype=torch.bfloat16); blk = torch.randn(N, HD, device=dev, dtype=torch.bfloat16)
    inj = torch.randn(N, HCN, device=dev, dtype=torch.bfloat16); w = torch.randn(HD, device=dev, dtype=torch.bfloat16) * 0.1
    x = torch.randn(N, DIM, device=dev, dtype=torch.bfloat16); g = torch.randn(N, DIM, device=dev, dtype=torch.bfloat16)
    B = 2 * N * DIM  # bytes of one [N, DIM] bf16 tensor
    print(f"== N={N}: [N,DIM] bf16 = {B/1e6:.0f} MB", flush=True)
    t = timeit(lambda: torch.add(res, res)); print(f"  torch add [N,DIM]+[N,DIM]->[N,DIM]  (3 tensors): {t:8.1f} us  {3*B/t/1e3:6.0f} GB/s  <- achievable floor", flush=True)
    out0, y0 = HC._hc_combine_norm(res, blk, inj, w, EPS, HCN)
    t0 = timeit(lambda: HC._hc_combine_norm(res, blk, inj, w, EPS, HCN))
    traffic = B + B + B + 2*N*HD  # res read, out write, y write, block read once
    print(f"  combine_norm stock : {t0:8.1f} us  {traffic/t0/1e3:6.0f} GB/s (block counted once)", flush=True)
    for BLOCK, nw in ((512,4),(1024,4),(1024,8),(2560,8)):
        try:
            o, y = combine_norm_v2(res, blk, inj, w, EPS, HCN, BLOCK=BLOCK, num_warps=nw)
            t1 = timeit(lambda: combine_norm_v2(res, blk, inj, w, EPS, HCN, BLOCK=BLOCK, num_warps=nw))
            print(f"  combine_norm v2 BLOCK={BLOCK} warps={nw}: {t1:8.1f} us  {traffic/t1/1e3:6.0f} GB/s  x{t0/t1:4.2f}  max|d out|={(o.float()-out0.float()).abs().max():.4f} max|d y|={(y.float()-y0.float()).abs().max():.4f}", flush=True)
        except Exception as ex: print(f"  combine_norm v2 BLOCK={BLOCK} warps={nw}: ERR {str(ex)[:100]}", flush=True)
    m0 = HC._hc_gate_mix(x, g, HCN); t0 = timeit(lambda: HC._hc_gate_mix(x, g, HCN)); traffic = 2*B + 2*N*HD
    print(f"  gate_mix stock : {t0:8.1f} us  {traffic/t0/1e3:6.0f} GB/s", flush=True)
    for ROWS, BLOCK, nw in ((1,512,4),(4,512,4),(4,1024,8),(8,512,8),(16,256,8)):
        try:
            m = gate_mix_v2(x, g, HCN, ROWS=ROWS, BLOCK=BLOCK, num_warps=nw); t1 = timeit(lambda: gate_mix_v2(x, g, HCN, ROWS=ROWS, BLOCK=BLOCK, num_warps=nw))
            print(f"  gate_mix v2 ROWS={ROWS} BLOCK={BLOCK} warps={nw}: {t1:8.1f} us  {traffic/t1/1e3:6.0f} GB/s  x{t0/t1:4.2f}  max|d|={(m.float()-m0.float()).abs().max():.4f}", flush=True)
        except Exception as ex: print(f"  gate_mix v2 ROWS={ROWS} BLOCK={BLOCK}: ERR {str(ex)[:100]}", flush=True)
