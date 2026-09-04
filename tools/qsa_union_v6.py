#!/usr/bin/env python3
"""Tile-union QSA sparse attention, v6 standalone = v5 (qsa_union_v3.py) with two kernel switches to bisect the R=4
regression of finding 107: MEMB_BITS (R-bit mask per union block, [BNB] load) vs int8 membership matrix ([M, BNB] load);
PREPHYS (physical token base pre-resolved per union block) vs in-kernel page-table lookup. Usage: qsa_union_v6.py <sel_*.pt>
Original v5 header:
 1. union built straight from the 512 block ids per row (exactly R*512 sort inputs), tail tokens from query positions;
 2. membership packed as R bits per union block (uni_mask int32 via atomic_or), tested in-kernel from a [BNB] load;
 3. page translation pre-resolved per union block in the precompute: uni_phys = physical_page*PAGE + base_offset
    (-1 if invalid), so the attention loop only forms phys+j and loads K/V; tails pre-resolved the same way;
 5. R x BNB x warps sweep. Reports precompute, kernel, total vs stock and max|diff|. Usage: qsa_union_v3.py <sel_*.pt>"""
import sys, statistics, torch, triton, triton.language as tl
import vllm  # noqa
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
dev="cuda"; torch.manual_seed(0); f=sys.argv[1]
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
KB = TOPK//CR; G = HQ//HKV; GP = 16; TAIL_COLS = 16
assert PAGE % CR == 0
d=torch.load(f); blocks=d["blocks"].to(dev).to(torch.int32).contiguous(); vis=d["visible_blocks"].to(dev).to(torch.int32)
rows=blocks.shape[0]; qpos=(vis*CR + torch.arange(rows, device=dev) % CR).to(torch.int32)
logical=torch.empty((rows, TOPK + CR - 1), device=dev, dtype=torch.int32); expand_qsa_block_indices(blocks, qpos, vis, CR, TOPK, logical)
ctx=int(qpos.max().item())+1; npages=(ctx+PAGE-1)//PAGE; NREQ=3; REQ=1
block_table=torch.randperm(npages*NREQ, device=dev, dtype=torch.int32).view(NREQ, npages).contiguous()
q=(torch.randn(rows, HQ, D, device=dev)*0.2).to(torch.bfloat16)
kc=(torch.randn(npages*NREQ, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16); vc=(torch.randn(npages*NREQ, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16)
t2r=torch.full((rows,), REQ, device=dev, dtype=torch.int32)
def timeit(fn, n=5, reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)
out_ref=torch.empty_like(q); Q.qsa_sparse_paged_attention(q, kc, vc, logical, block_table, t2r, out_ref)
t_ref=timeit(lambda: Q.qsa_sparse_paged_attention(q, kc, vc, logical, block_table, t2r, out_ref))
print(f"{f.split('/')[-1]}: rows={rows} ctx={ctx} stock {t_ref:8.1f} us", flush=True)

@triton.jit
def _build(sorted_ptr, blk_ptr, mask_ptr, cnt_ptr, stride_sorted, stride_u, N: tl.constexpr):
    t=tl.program_id(0); i=tl.arange(0, N)
    packed=tl.load(sorted_ptr + t*stride_sorted + i); prev=tl.load(sorted_ptr + t*stride_sorted + i - 1, mask=i > 0, other=-8)
    blk=packed//8; r=packed%8; valid=blk < (1 << 27); first=(blk != prev//8) & valid
    pos=tl.cumsum(first.to(tl.int32)) - 1
    tl.store(blk_ptr + t*stride_u + pos, blk, mask=first)
    tl.atomic_or(mask_ptr + t*stride_u + pos, (1 << r).to(tl.int32), mask=valid)
    tl.store(cnt_ptr + t, tl.sum(first.to(tl.int32)))

def build(blocks, qpos, R, req):
    rows=blocks.shape[0]; T=(rows+R-1)//R; N=R*KB; pad=T*R-rows
    b=torch.cat([blocks, torch.full((pad, KB), -1, device=dev, dtype=torch.int32)]) if pad else blocks
    b=b.view(T, R, KB); rr=torch.arange(R, device=dev, dtype=torch.int32)[None, :, None]
    packed=torch.where(b >= 0, b*8 + rr, torch.full_like(b, (1 << 27)*8 + 7)).view(T, N); packed,_=torch.sort(packed, dim=1)
    ublk=torch.full((T, N), -1, device=dev, dtype=torch.int32); umask=torch.zeros((T, N), device=dev, dtype=torch.int32); cnt=torch.empty(T, device=dev, dtype=torch.int32)
    _build[(T,)](packed, ublk, umask, cnt, packed.stride(0), ublk.stride(0), N=N, num_warps=4)
    # item 3: pre-resolve union blocks to physical token bases (request's page table row)
    bpp=PAGE//CR; bt=block_table[req]; lp=(ublk.clamp(min=0)//bpp); ok=(ublk >= 0) & (lp < bt.shape[0])
    pp=bt[lp.clamp(max=bt.shape[0]-1)]; ok &= pp >= 0
    uphys=torch.where(ok, pp*PAGE + (ublk.clamp(min=0) % bpp)*CR, torch.full_like(ublk, -1)).to(torch.int32)
    # tails from query positions (item 1): tail_start..qpos, <= CR-1 tokens, pre-resolved
    tail_start=((qpos + 1)//CR)*CR; tcount=(qpos + 1) - tail_start
    j=torch.arange(CR-1, device=dev, dtype=torch.int32)[None, :]; ttok=torch.where(j < tcount[:, None], tail_start[:, None] + j, torch.full_like(j.expand(rows, -1), -1))
    tp=ttok.clamp(min=0)//PAGE; tok_ok=(ttok >= 0) & (tp < bt.shape[0]); tpp=bt[tp.clamp(max=bt.shape[0]-1)]; tok_ok &= tpp >= 0
    tphys=torch.where(tok_ok, tpp*PAGE + ttok.clamp(min=0) % PAGE, torch.full_like(ttok, -1)).to(torch.int32)
    tails=torch.full((T, TAIL_COLS), -1, device=dev, dtype=torch.int32)
    tp_=torch.cat([tphys, torch.full((pad, CR-1), -1, device=dev, dtype=torch.int32)]) if pad else tphys
    tails[:, : R*(CR-1)]=tp_.view(T, R*(CR-1))
    mem=((umask[:, None, :] >> torch.arange(R, device=dev, dtype=torch.int32)[None, :, None]) & 1).to(torch.int8).contiguous()
    return uphys, umask, cnt, tails, T, ublk, mem

@triton.jit
def _attn(q_ptr, k_ptr, v_ptr, uphys_ptr, umask_ptr, cnt_ptr, tail_ptr, out_ptr, mem_ptr, bt_ptr, stride_q_row, stride_q_head, stride_k_token, stride_k_head,
          stride_u, stride_tail, stride_out_row, stride_out_head, stride_mem_t, stride_mem_r, num_rows, table_width, num_pages,
          R: tl.constexpr, GP: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BNB: tl.constexpr, CR: tl.constexpr, TAIL_COLS: tl.constexpr,
          PAGE: tl.constexpr, MEMB_BITS: tl.constexpr, PREPHYS: tl.constexpr):
    BPP: tl.constexpr = PAGE // CR
    tile=tl.program_id(0); kv=tl.program_id(1)
    M: tl.constexpr = R*GP; BN: tl.constexpr = BNB*CR; TPR: tl.constexpr = CR-1
    m_off=tl.arange(0, M); r_of_m=m_off//GP; h_of_m=m_off%GP; d_off=tl.arange(0, D); b_off=tl.arange(0, BNB); j_off=tl.arange(0, CR)
    row=tile*R + r_of_m; qmask=(row < num_rows) & (h_of_m < G)
    q=tl.load(q_ptr + row[:, None]*stride_q_row + (kv*G + h_of_m[:, None])*stride_q_head + d_off[None, :], mask=qmask[:, None], other=0.0)
    mx=tl.full((M,), -1.0e20, tl.float32); nrm=tl.zeros((M,), tl.float32); acc=tl.zeros((M, D), tl.float32)
    scale: tl.constexpr = (D ** -0.5) * 1.4426950408889634
    rbit=(1 << r_of_m).to(tl.int32)
    ubound=tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        em=(t + b_off) < ubound
        if PREPHYS:
            phys=tl.load(uphys_ptr + tile*stride_u + t + b_off, mask=em, other=-1)          # [BNB] physical token base
        else:
            blk=tl.load(uphys_ptr + tile*stride_u + t + b_off, mask=em, other=-1)           # [BNB] logical block id
            lp=tl.maximum(blk, 0) // BPP; ok=(blk >= 0) & (lp < table_width)
            pp=tl.load(bt_ptr + tl.minimum(lp, table_width - 1), mask=ok, other=-1); ok &= (pp >= 0) & (pp < num_pages)
            phys=tl.where(ok, pp*PAGE + (tl.maximum(blk, 0) % BPP)*CR, -1)
        tok2=tl.where((phys >= 0)[:, None], phys[:, None] + j_off[None, :], -1)
        tok=tl.reshape(tok2, (BN,)); valid=tok >= 0; st=tl.maximum(tok, 0).to(tl.int64)
        keys=tl.load(k_ptr + st[None, :]*stride_k_token + kv*stride_k_head + d_off[:, None], mask=valid[None, :], other=0.0)
        vals=tl.load(v_ptr + st[:, None]*stride_k_token + kv*stride_k_head + d_off[None, :], mask=valid[:, None], other=0.0)
        if MEMB_BITS:
            um=tl.load(umask_ptr + tile*stride_u + t + b_off, mask=em, other=0)             # [BNB] R-bit membership
            memb=((um[None, :] & rbit[:, None]) != 0)                                      # [M, BNB]
        else:
            memb=tl.load(mem_ptr + tile*stride_mem_t + r_of_m[:, None]*stride_mem_r + t + b_off[None, :], mask=em[None, :], other=0) > 0
        memt=tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN)) & valid[None, :]
        s=tl.dot(q, keys)*scale; s=tl.where(memt, s, -1.0e20)
        nmx=tl.maximum(mx, tl.max(s, 1)); a=tl.math.exp2(mx - nmx); p=tl.where(memt, tl.math.exp2(s - nmx[:, None]), 0.0)
        acc=tl.dot(p.to(vals.dtype), vals, acc=acc*a[:, None]); nrm=nrm*a + tl.sum(p, 1); mx=nmx
    tt=tl.arange(0, TAIL_COLS); slot_row=tt//TPR
    tph=tl.load(tail_ptr + tile*stride_tail + tt, mask=tt < R*TPR, other=-1); valid=tph >= 0; st=tl.maximum(tph, 0).to(tl.int64)
    keys=tl.load(k_ptr + st[None, :]*stride_k_token + kv*stride_k_head + d_off[:, None], mask=valid[None, :], other=0.0)
    vals=tl.load(v_ptr + st[:, None]*stride_k_token + kv*stride_k_head + d_off[None, :], mask=valid[:, None], other=0.0)
    act=(r_of_m[:, None] == slot_row[None, :]) & valid[None, :]
    s=tl.dot(q, keys)*scale; s=tl.where(act, s, -1.0e20)
    nmx=tl.maximum(mx, tl.max(s, 1)); a=tl.math.exp2(mx - nmx); p=tl.where(act, tl.math.exp2(s - nmx[:, None]), 0.0)
    acc=tl.dot(p.to(vals.dtype), vals, acc=acc*a[:, None]); nrm=nrm*a + tl.sum(p, 1)
    has=nrm > 0; o=tl.where(has[:, None], acc/tl.maximum(nrm[:, None], 1.0e-20), 0.0)
    tl.store(out_ptr + row[:, None]*stride_out_row + (kv*G + h_of_m[:, None])*stride_out_head + d_off[None, :], o.to(tl.bfloat16), mask=qmask[:, None])

# k_cache [blocks, PAGE, kv, D] contiguous -> token stride = kv*D, physical token index = block*PAGE + off
assert kc.stride(0) == PAGE*kc.stride(1)
btrow=block_table[REQ].contiguous()
for R, BNB, warps in ((4, 16, 8), (4, 16, 4), (4, 8, 8), (2, 8, 4), (2, 16, 4)):
    uphys, umask, cnt, tails, T, ublk, mem = build(blocks, qpos, R, REQ); torch.cuda.synchronize()
    print(f"  R={R} BNB={BNB} warps={warps}: union mean {float(cnt.float().mean()):.0f} blocks/tile", flush=True)
    for MB in (1, 0):
        for PP in (1, 0):
            try:
                def run():
                    out=torch.empty_like(q)
                    _attn[(T, HKV)](q, kc, vc, uphys if PP else ublk, umask, cnt, tails, out, mem, btrow, q.stride(0), q.stride(1), kc.stride(1), kc.stride(2), uphys.stride(0), tails.stride(0), out.stride(0), out.stride(1),
                                    mem.stride(0), mem.stride(1), rows, btrow.shape[0], kc.shape[0],
                                    R=R, GP=GP, G=G, D=D, BNB=BNB, CR=CR, TAIL_COLS=TAIL_COLS, PAGE=PAGE, MEMB_BITS=MB, PREPHYS=PP, num_warps=warps, num_stages=1)
                    return out
                out=run(); torch.cuda.synchronize(); diff=(out.float()-out_ref.float()).abs().max(); t=timeit(run)
                print(f"    bits={MB} prephys={PP}: kernel {t:8.1f} us -> kernel x{t_ref/t:4.2f}  max|diff|={float(diff):.4f}", flush=True)
            except Exception as ex: print(f"    bits={MB} prephys={PP}: ERR {str(ex).splitlines()[0][:110]}", flush=True)
