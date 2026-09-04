#!/usr/bin/env python3
"""Tile-union QSA sparse attention, v2 (finding 103 -> implementation step 1): union at BLOCK granularity
(512 ids/row, compress ratio 4), precompute = torch.sort over each tile's concatenated block ids + one Triton
kernel (first-occurrence flags, cumsum positions, scatter of union ids and per-row membership), attention kernel
loops over union blocks (16 blocks = 64 tokens per iteration) and expands tokens itself. Reports precompute time,
kernel time and max|diff| vs vLLM's stock kernel on a real selection dump. Usage: qsa_union_v2.py <sel_*.pt> [R]"""
import sys, statistics, torch, triton, triton.language as tl
import vllm  # noqa
try: from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention
except ImportError: from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
dev="cuda"; torch.manual_seed(0)
f=sys.argv[1]; R=int(sys.argv[2]) if len(sys.argv)>2 else 4
HQ, HKV, D, PAGE, CR, KB = 24, 2, 256, 1600, 4, 512     # KB = blocks per row, TOPK = KB*CR = 2048
G = HQ//HKV; GP = 16; BNB = 16; BN = BNB*CR              # 16 union blocks = 64 tokens per iteration
d=torch.load(f); blocks=d["blocks"].to(dev).to(torch.int32).contiguous()   # [rows, 512], -1 padded (front)
rows=blocks.shape[0]; ctx=int(blocks.max().item()+1)*CR
tok=(blocks[:, :, None]*CR + torch.arange(CR, device=dev)[None, None, :]).reshape(rows, -1)
tok=torch.where(blocks.repeat_interleave(CR, dim=1) >= 0, tok, torch.full_like(tok, -1)).to(torch.int32).contiguous()
npages=(ctx+PAGE-1)//PAGE
q=(torch.randn(rows, HQ, D, device=dev)*0.2).to(torch.bfloat16)
kc=(torch.randn(npages, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16); vc=(torch.randn(npages, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16)
block_table=torch.arange(npages, device=dev, dtype=torch.int32)[None, :].contiguous(); token_to_req=torch.zeros(rows, device=dev, dtype=torch.int32)
def timeit(fn, n=5, reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)
print(f"{f.split('/')[-1]}: rows={rows} ctx={ctx} R={R}", flush=True)
out_ref=qsa_sparse_paged_attention(q, kc, vc, tok, block_table, token_to_req); torch.cuda.synchronize()
t_ref=timeit(lambda: qsa_sparse_paged_attention(q, kc, vc, tok, block_table, token_to_req))

# ---------------- union precompute ----------------
@triton.jit
def _union_build_kernel(sorted_ptr, uni_ptr, mem_ptr, cnt_ptr, stride_sorted, stride_uni, stride_mem_t, stride_mem_r,
                        N: tl.constexpr, R: tl.constexpr, UB: tl.constexpr):
    # sorted_ptr: [T, N] packed = id*8 + r (ids ascending, BIG for -1); one program per tile
    t=tl.program_id(0); i=tl.arange(0, N)
    packed=tl.load(sorted_ptr + t*stride_sorted + i)
    prev=tl.load(sorted_ptr + t*stride_sorted + i - 1, mask=i > 0, other=-8)     # previous element (any row)
    ids=packed // 8; r=packed % 8; pid=prev // 8
    valid=ids < (1 << 26)
    first=(ids != pid) & valid
    pos=tl.cumsum(first.to(tl.int32)) - 1                                        # 0-based union position
    tl.store(uni_ptr + t*stride_uni + pos, ids, mask=first)
    tl.store(mem_ptr + t*stride_mem_t + r*stride_mem_r + pos, tl.full((N,), 1, tl.int8), mask=valid)
    tl.store(cnt_ptr + t, tl.sum(first.to(tl.int32)))

def build_union(blocks, R):
    rows=blocks.shape[0]; T=(rows+R-1)//R; pad=T*R-rows
    b=torch.cat([blocks, torch.full((pad, KB), -1, device=dev, dtype=torch.int32)]) if pad else blocks
    b=b.view(T, R, KB)
    rr=torch.arange(R, device=dev, dtype=torch.int32)[None, :, None]
    packed=torch.where(b >= 0, b*8 + rr, torch.full_like(b, (1 << 26)*8 + 7)).view(T, R*KB)
    packed,_=torch.sort(packed, dim=1)
    UB=R*KB                                                                        # worst case: disjoint rows
    uni=torch.full((T, UB), -1, device=dev, dtype=torch.int32); mem=torch.zeros((T, R, UB), device=dev, dtype=torch.int8); cnt=torch.empty(T, device=dev, dtype=torch.int32)
    _union_build_kernel[(T,)](packed, uni, mem, cnt, packed.stride(0), uni.stride(0), mem.stride(0), mem.stride(1), N=R*KB, R=R, UB=UB, num_warps=4)
    return uni, mem, cnt, T, UB
uni, mem, cnt, T, UB = build_union(blocks, R); torch.cuda.synchronize()
t_pre=timeit(lambda: build_union(blocks, R), n=5, reps=5)
# sanity: membership count == selected blocks; union ids per tile == unique of the tile's rows
assert int(mem.sum()) == int((blocks >= 0).sum()), (int(mem.sum()), int((blocks >= 0).sum()))
t0=0; chk=torch.unique(blocks[t0*R:(t0+1)*R][blocks[t0*R:(t0+1)*R] >= 0]); assert torch.equal(uni[t0, :int(cnt[t0])], chk.to(torch.int32)), "union mismatch tile 0"
print(f"  union blocks: tiles={T} mean count={float(cnt.float().mean()):.0f} (x{float(cnt.float().mean())/max(1,float((blocks>=0).sum(1).float().mean())):.2f} of a row) max={int(cnt.max())}  precompute {t_pre:8.1f} us", flush=True)

# ---------------- attention kernel over union blocks ----------------
@triton.jit
def _union_attn_kernel(q_ptr, kc_ptr, vc_ptr, uni_ptr, mem_ptr, cnt_ptr, bt_ptr, out_ptr, stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
                       stride_uni, stride_mem_t, stride_mem_r, stride_out_row, stride_out_head, num_rows,
                       R: tl.constexpr, GP: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BNB: tl.constexpr, CR: tl.constexpr, PAGE: tl.constexpr):
    tile=tl.program_id(0); kv=tl.program_id(1)
    M: tl.constexpr = R*GP; BN: tl.constexpr = BNB*CR
    m_off=tl.arange(0, M); r_of_m=m_off//GP; h_of_m=m_off%GP; d_off=tl.arange(0, D)
    b_off=tl.arange(0, BNB); j_off=tl.arange(0, CR)
    row=tile*R + r_of_m; qmask=(row < num_rows) & (h_of_m < G)
    q=tl.load(q_ptr + row[:, None]*stride_q_row + (kv*G + h_of_m[:, None])*stride_q_head + d_off[None, :], mask=qmask[:, None], other=0.0)
    mx=tl.full((M,), -1.0e20, tl.float32); nrm=tl.zeros((M,), tl.float32); acc=tl.zeros((M, D), tl.float32)
    scale: tl.constexpr = (D ** -0.5) * 1.4426950408889634
    ubound=tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        ub=tl.load(uni_ptr + tile*stride_uni + t + b_off, mask=(t + b_off) < ubound, other=-1)          # [BNB] union block ids
        tok2=ub[:, None]*CR + j_off[None, :]                                                           # [BNB, CR]
        tok=tl.reshape(tok2, (BN,)); valid=tok >= 0; st=tl.maximum(tok, 0); page=st//PAGE; off=st%PAGE
        ppage=tl.load(bt_ptr + page, mask=valid, other=0)
        keys=tl.load(kc_ptr + (ppage[None, :]*stride_k_block + off[None, :]*stride_k_token + kv*stride_k_head) + d_off[:, None], mask=valid[None, :], other=0.0)
        vals=tl.load(vc_ptr + (ppage[:, None]*stride_k_block + off[:, None]*stride_k_token + kv*stride_k_head) + d_off[None, :], mask=valid[:, None], other=0.0)
        memb=tl.load(mem_ptr + tile*stride_mem_t + r_of_m[:, None]*stride_mem_r + t + b_off[None, :], mask=((t + b_off) < ubound)[None, :], other=0)   # [M, BNB] int8
        memt=tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN))                       # [M, BN]
        s=tl.dot(q, keys)*scale
        s=tl.where(memt > 0, s, -1.0e20)
        nmx=tl.maximum(mx, tl.max(s, 1)); a=tl.math.exp2(mx - nmx); p=tl.where(memt > 0, tl.math.exp2(s - nmx[:, None]), 0.0)
        acc=tl.dot(p.to(vals.dtype), vals, acc=acc*a[:, None]); nrm=nrm*a + tl.sum(p, 1); mx=nmx
    has=nrm > 0; o=tl.where(has[:, None], acc/tl.maximum(nrm[:, None], 1.0e-20), 0.0)
    tl.store(out_ptr + row[:, None]*stride_out_row + (kv*G + h_of_m[:, None])*stride_out_head + d_off[None, :], o.to(tl.bfloat16), mask=qmask[:, None])

def run_union(warps=8):
    out=torch.empty_like(q)
    _union_attn_kernel[(T, HKV)](q, kc, vc, uni, mem, cnt, block_table, out, q.stride(0), q.stride(1), kc.stride(0), kc.stride(1), kc.stride(2),
                                 uni.stride(0), mem.stride(0), mem.stride(1), out.stride(0), out.stride(1), rows,
                                 R=R, GP=GP, G=G, D=D, BNB=BNB, CR=CR, PAGE=PAGE, num_warps=warps, num_stages=1)
    return out
for warps in (8, 4):
    try:
        out=run_union(warps); torch.cuda.synchronize(); diff=(out.float()-out_ref.float()).abs().max(); t=timeit(lambda: run_union(warps))
        print(f"  union-v2 R={R} warps={warps}: kernel {t:8.1f} us + precompute {t_pre:6.1f} us = {t+t_pre:8.1f}  vs stock {t_ref:8.1f} us -> x{t_ref/(t+t_pre):4.2f} (kernel alone x{t_ref/t:4.2f})  max|diff|={float(diff):.4f}", flush=True)
    except Exception as ex: print(f"  union-v2 warps={warps}: ERR {str(ex).splitlines()[0][:140]}", flush=True)
