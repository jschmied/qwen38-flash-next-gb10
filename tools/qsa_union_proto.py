#!/usr/bin/env python3
"""Tile-union QSA sparse attention prototype (plan §5 item 3, findings 92/96). Consumes a real selection dump
(qsadump2: block ids per query row, compress ratio 4 -> token = block*4 + j), builds synthetic q / paged K,V at the
model's geometry (24 q heads, 2 kv heads, head_dim 256, page 1600), runs the STOCK kernel (vLLM's
qsa_sparse_paged_attention) and the UNION kernel (R consecutive rows per program share one gathered token set;
per-row membership mask; dot M = R*16), and reports max|diff| and timings (union precompute reported separately).
Usage: qsa_union_proto.py <sel_*.pt> [R=4] [BN=64]"""
import sys, time, statistics, torch, triton, triton.language as tl
import vllm  # noqa
try: from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import qsa_sparse_paged_attention
except ImportError: from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention
dev="cuda"; torch.manual_seed(0)
f=sys.argv[1]; R=int(sys.argv[2]) if len(sys.argv)>2 else 4; BN=int(sys.argv[3]) if len(sys.argv)>3 else 64
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
G = HQ // HKV; GP = 16  # group of 12 heads padded to 16 rows
d=torch.load(f); blocks=d["blocks"].to(dev)                    # [rows, 512] block ids, -1 padded
rows=blocks.shape[0]; ctx=int(blocks.max().item()+1)*CR
# token indices [rows, TOPK] from blocks (block*4 + j), -1 kept
tok=(blocks[:, :, None]*CR + torch.arange(CR, device=dev)[None, None, :]).reshape(rows, -1)
tok=torch.where(blocks.repeat_interleave(CR, dim=1) >= 0, tok, torch.full_like(tok, -1)).to(torch.int32).contiguous()
npages=(ctx + PAGE - 1)//PAGE
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
print(f"{f.split('/')[-1]}: rows={rows} ctx={ctx} pages={npages} R={R} BN={BN}", flush=True)
# ---- stock ----
out_ref=qsa_sparse_paged_attention(q, kc, vc, tok, block_table, token_to_req); torch.cuda.synchronize()
t_ref=timeit(lambda: qsa_sparse_paged_attention(q, kc, vc, tok, block_table, token_to_req))
# ---- union precompute (torch): per tile of R rows: sorted unique tokens (padded to BN multiple), membership [R, U] ----
def build_union(tok, R, BN):
    rows=tok.shape[0]; T=(rows+R-1)//R; pad=T*R-rows
    t=torch.cat([tok, torch.full((pad, tok.shape[1]), -1, device=dev, dtype=tok.dtype)]) if pad else tok
    t=t.view(T, R*tok.shape[1])
    s,_=torch.sort(t, dim=1)                                    # -1 first
    first=torch.ones_like(s, dtype=torch.bool); first[:,1:]=s[:,1:]!=s[:,:-1]; first &= s>=0
    counts=first.sum(1); U=int(counts.max().item()); U=(U+BN-1)//BN*BN
    pos=torch.cumsum(first.to(torch.int32), dim=1)-1
    uni=torch.full((T, U), -1, device=dev, dtype=torch.int32)
    ti=torch.arange(T, device=dev)[:, None].expand_as(s)
    uni[ti[first], pos[first]]=s[first]
    # membership: token of row r present in union position p
    member=torch.zeros((T, R, U), device=dev, dtype=torch.uint8)
    tr=t.view(T, R, tok.shape[1])
    # for each (tile,row,token>=0): position via searchsorted on uni
    valid=tr>=0
    key=torch.where(uni >= 0, uni, torch.full_like(uni, 2**31-1))   # sorted ascending, padding last
    p=torch.searchsorted(key.contiguous(), tr.reshape(T, -1).contiguous()).view(T, R, -1).clamp(max=U-1)
    ok=valid & (torch.gather(uni, 1, p.view(T, -1)).view(T, R, -1)==tr)
    tt=torch.arange(T, device=dev)[:, None, None].expand_as(p); rr=torch.arange(R, device=dev)[None, :, None].expand_as(p)
    member[tt[ok], rr[ok], p[ok]]=1
    ucount=((counts + BN - 1)//BN*BN).to(torch.int32).contiguous()     # per-tile column bound (multiple of BN)
    return uni.contiguous(), member.contiguous(), T, U, ucount
torch.cuda.synchronize(); t0=time.perf_counter(); uni, member, T, U, ucount = build_union(tok, R, BN); torch.cuda.synchronize(); t_pre=(time.perf_counter()-t0)*1e6
mean_sel=float((tok>=0).sum(1).float().mean()); print(f"  union: tiles={T} U(max,padded)={U} mean per-tile columns={float(ucount.float().mean()):.0f} mean|sel|={mean_sel:.0f}  (precompute torch {t_pre/1e3:.1f} ms)", flush=True)
# membership sanity: every selected token of every row must be present exactly once
assert int(member.sum()) == int((tok>=0).sum()), (int(member.sum()), int((tok>=0).sum()))

@triton.jit
def _union_kernel(q_ptr, kc_ptr, vc_ptr, uni_ptr, mem_ptr, ucount_ptr, bt_ptr, out_ptr, stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
                  stride_uni, stride_mem_t, stride_mem_r, stride_out_row, stride_out_head, num_rows,
                  R: tl.constexpr, GP: tl.constexpr, G: tl.constexpr, D: tl.constexpr, U: tl.constexpr, BN: tl.constexpr, PAGE: tl.constexpr):
    tile=tl.program_id(0); kv=tl.program_id(1)
    M: tl.constexpr = R*GP
    m_off=tl.arange(0, M); r_of_m=m_off//GP; h_of_m=m_off%GP; d_off=tl.arange(0, D); c_off=tl.arange(0, BN)
    row=tile*R + r_of_m; qmask=(row < num_rows) & (h_of_m < G)
    q=tl.load(q_ptr + row[:, None]*stride_q_row + (kv*G + h_of_m[:, None])*stride_q_head + d_off[None, :], mask=qmask[:, None], other=0.0)
    mx=tl.full((M,), -1.0e20, tl.float32); nrm=tl.zeros((M,), tl.float32); acc=tl.zeros((M, D), tl.float32)
    scale: tl.constexpr = (D ** -0.5) * 1.4426950408889634
    ubound=tl.load(ucount_ptr + tile)
    for t in range(0, ubound, BN):
        cols=t + c_off
        tok=tl.load(uni_ptr + tile*stride_uni + cols)                       # -1 padded
        valid=tok >= 0; st=tl.maximum(tok, 0); page=st//PAGE; off=st%PAGE
        ppage=tl.load(bt_ptr + page, mask=valid, other=0)
        keys=tl.load(kc_ptr + (ppage[None, :]*stride_k_block + off[None, :]*stride_k_token + kv*stride_k_head) + d_off[:, None], mask=valid[None, :], other=0.0)
        vals=tl.load(vc_ptr + (ppage[:, None]*stride_k_block + off[:, None]*stride_k_token + kv*stride_k_head) + d_off[None, :], mask=valid[:, None], other=0.0)
        mem=tl.load(mem_ptr + tile*stride_mem_t + r_of_m[:, None]*stride_mem_r + cols[None, :], mask=valid[None, :], other=0)  # [M, BN] uint8
        s=tl.dot(q, keys)*scale
        s=tl.where(mem > 0, s, -1.0e20)
        nmx=tl.maximum(mx, tl.max(s, 1)); a=tl.math.exp2(mx - nmx); p=tl.where(mem > 0, tl.math.exp2(s - nmx[:, None]), 0.0)
        acc=tl.dot(p.to(vals.dtype), vals, acc=acc*a[:, None]); nrm=nrm*a + tl.sum(p, 1); mx=nmx
    has=nrm > 0; o=tl.where(has[:, None], acc/tl.maximum(nrm[:, None], 1.0e-20), 0.0)
    tl.store(out_ptr + row[:, None]*stride_out_row + (kv*G + h_of_m[:, None])*stride_out_head + d_off[None, :], o.to(tl.bfloat16), mask=qmask[:, None])

def run_union(warps=8, stages=1):
    out=torch.empty_like(q)
    _union_kernel[(T, HKV)](q, kc, vc, uni, member, ucount, block_table, out, q.stride(0), q.stride(1), kc.stride(0), kc.stride(1), kc.stride(2),
                            uni.stride(0), member.stride(0), member.stride(1), out.stride(0), out.stride(1), rows,
                            R=R, GP=GP, G=G, D=D, U=U, BN=BN, PAGE=PAGE, num_warps=warps, num_stages=stages)
    return out
for warps, stages in ((8,1),(4,1),(8,2)):
    try:
        out=run_union(warps, stages); torch.cuda.synchronize()
        diff=(out.float()-out_ref.float()).abs(); rel=diff.max()/out_ref.float().abs().max()
        t=timeit(lambda: run_union(warps, stages))
        print(f"  union R={R} BN={BN} warps={warps} stages={stages}: {t:9.1f} us  vs stock {t_ref:9.1f} us  -> x{t_ref/t:4.2f}  ({t/rows:6.2f} vs {t_ref/rows:6.2f} us/row)  max|diff|={diff.max():.4f} rel={rel:.2e}", flush=True)
    except Exception as ex: print(f"  union warps={warps} stages={stages}: ERR {str(ex).splitlines()[0][:120]}", flush=True)
