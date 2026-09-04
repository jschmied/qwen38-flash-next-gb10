"""Tile-union QSA sparse attention for prefill (findings 92/96/103/104), env-gated: VLLM_QSA_UNION=1 (default off).
Appends the union precompute + attention kernels to vllm/models/qwen4_exp/nvidia/ops/qsa.py and routes
`qsa_sparse_paged_attention` through them for single-request prefill batches (>= 256 rows). Union entries are
(token*2 + flag): flag 0 = a whole compressed block (expanded to CR tokens in-kernel), flag 1 = one causal-tail
token of the query's open block (never expanded). R chosen per call by the finding-104 cost model.
Target: ops/qsa.py of the running interpreter (VLLM_QSA_OPS_PY overrides). `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_QSA_OPS_PY"): return os.environ["VLLM_QSA_OPS_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/ops/qsa.py")
TARGET = _target()
MARK = "# ---- QSA UNION (jschmied 2026-09-04) ----"
ADD = MARK + '''
import os as _os
import torch
import triton
import triton.language as tl

_QSA_UNION = _os.environ.get("VLLM_QSA_UNION", "0") not in ("0", "", "false", "False")
_QSA_UNION_MIN_ROWS = int(_os.environ.get("VLLM_QSA_UNION_MIN_ROWS", "256"))
_QSA_UNION_C4, _QSA_UNION_C2 = 12.8, 9.1  # ns per (tile, union entry), finding 104 cost model
if _QSA_UNION:
    print("QSAUNION active", flush=True)


@triton.jit
def _qsa_union_build_kernel(sorted_ptr, uni_ptr, mem_ptr, cnt_ptr, stride_sorted, stride_uni,
                            stride_mem_t, stride_mem_r, N: tl.constexpr):
    # sorted_ptr[t]: ascending packed entries (entry*8 + row_in_tile), BIG*8+7 for padding
    t = tl.program_id(0)
    i = tl.arange(0, N)
    packed = tl.load(sorted_ptr + t * stride_sorted + i)
    prev = tl.load(sorted_ptr + t * stride_sorted + i - 1, mask=i > 0, other=-8)
    entry = packed // 8
    r = packed % 8
    valid = entry < (1 << 27)
    first = (entry != prev // 8) & valid
    pos = tl.cumsum(first.to(tl.int32)) - 1
    tl.store(uni_ptr + t * stride_uni + pos, entry, mask=first)
    tl.store(mem_ptr + t * stride_mem_t + r * stride_mem_r + pos, tl.full((N,), 1, tl.int8), mask=valid)
    tl.store(cnt_ptr + t, tl.sum(first.to(tl.int32)))


@triton.jit
def _qsa_union_attn_kernel(q_ptr, k_cache_ptr, v_cache_ptr, uni_ptr, mem_ptr, cnt_ptr, block_table_ptr, out_ptr,
                           stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
                           stride_v_block, stride_v_token, stride_v_head, stride_uni, stride_mem_t, stride_mem_r,
                           stride_out_row, stride_out_head, num_rows, request, num_cache_blocks,
                           R: tl.constexpr, GP: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                           BNB: tl.constexpr, CR: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
                           STRIDE_TABLE_REQ: tl.constexpr):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    M: tl.constexpr = R * GP
    BN: tl.constexpr = BNB * CR
    m_off = tl.arange(0, M)
    r_of_m = m_off // GP
    h_of_m = m_off % GP
    dim_offsets = tl.arange(0, HEAD_DIM)
    b_off = tl.arange(0, BNB)
    j_off = tl.arange(0, CR)
    row = tile * R + r_of_m
    qmask = (row < num_rows) & (h_of_m < GROUP_SIZE)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(q_ptr + row[:, None] * stride_q_row + (first_head + h_of_m[:, None]) * stride_q_head
                    + dim_offsets[None, :], mask=qmask[:, None], other=0.0)
    max_value = tl.full((M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.zeros((M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    ubound = tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        emask = (t + b_off) < ubound
        entry = tl.load(uni_ptr + tile * stride_uni + t + b_off, mask=emask, other=-1)
        base = entry // 2
        is_block = (entry % 2) == 0
        tok2 = tl.where(is_block[:, None], base[:, None] + j_off[None, :],
                        tl.where(j_off[None, :] == 0, base[:, None], -1))
        tok2 = tl.where((entry >= 0)[:, None], tok2, -1)
        logical_token = tl.reshape(tok2, (BN,))
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(block_table_ptr + request * STRIDE_TABLE_REQ
                                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1), mask=valid, other=-1)
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
                       + kv_head * stride_k_head + dim_offsets[:, None], mask=valid[None, :], other=0.0)
        values = tl.load(v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                         + kv_head * stride_v_head + dim_offsets[None, :], mask=valid[:, None], other=0.0)
        memb = tl.load(mem_ptr + tile * stride_mem_t + r_of_m[:, None] * stride_mem_r + t + b_off[None, :],
                       mask=emask[None, :], other=0)
        memt = tl.reshape(tl.broadcast_to(memb[:, :, None], (M, BNB, CR)), (M, BN))
        active = (memt > 0) & valid[None, :]
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(active, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    has_values = normalizer > 0
    out = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    tl.store(out_ptr + row[:, None] * stride_out_row + (first_head + h_of_m[:, None]) * stride_out_head
             + dim_offsets[None, :], out.to(tl.bfloat16), mask=qmask[:, None])


def _qsa_union_entries(logical_indices: torch.Tensor, compress_ratio: int, token_topk: int) -> torch.Tensor:
    """[rows, block_topk + CR - 1] int32 entries: block starts (token*2) for the expanded prefix, tail
    tokens (token*2 + 1) after it, -1 elsewhere. Layout per expand_qsa_block_indices: complete blocks
    first (CR consecutive tokens each), then the causal tail, then -1."""
    rows = logical_indices.shape[0]
    block_topk = token_topk // compress_ratio
    cols = logical_indices[:, : block_topk * compress_ratio].reshape(rows, block_topk, compress_ratio)
    first = cols[:, :, 0]
    whole = (first >= 0) & ((first % compress_ratio) == 0) & (cols[:, :, compress_ratio - 1] == first + compress_ratio - 1)
    ent_blocks = torch.where(whole, first * 2, torch.full_like(first, -1))
    # the causal tail (<= CR-1 tokens) sits right after the last whole block: columns 4c .. 4c+CR-2
    c = whole.sum(dim=1, keepdim=True)
    width = logical_indices.shape[1]
    tcols = (c * compress_ratio + torch.arange(compress_ratio - 1, device=logical_indices.device)[None, :]).clamp(max=width - 1)
    tail = torch.gather(logical_indices, 1, tcols)
    ent_tail = torch.where(tail >= 0, tail * 2 + 1, torch.full_like(tail, -1))
    return torch.cat([ent_blocks, ent_tail], dim=1).to(torch.int32)


def _qsa_union_build(entries: torch.Tensor, R: int):
    rows, E = entries.shape
    T = (rows + R - 1) // R
    N = 1 << (R * E - 1).bit_length()
    dev = entries.device
    packed = torch.full((T, N), (1 << 27) * 8 + 7, device=dev, dtype=torch.int32)
    e = entries
    if T * R != rows:
        e = torch.cat([e, torch.full((T * R - rows, E), -1, device=dev, dtype=torch.int32)])
    e = e.view(T, R, E)
    rr = torch.arange(R, device=dev, dtype=torch.int32)[None, :, None]
    packed[:, : R * E] = torch.where(e >= 0, e * 8 + rr, torch.full_like(e, (1 << 27) * 8 + 7)).view(T, R * E)
    packed, _ = torch.sort(packed, dim=1)
    UB = R * E
    uni = torch.full((T, UB), -1, device=dev, dtype=torch.int32)
    mem = torch.zeros((T, R, UB), device=dev, dtype=torch.int8)
    cnt = torch.empty(T, device=dev, dtype=torch.int32)
    _qsa_union_build_kernel[(T,)](packed, uni, mem, cnt, packed.stride(0), uni.stride(0), mem.stride(0), mem.stride(1),
                                  N=N, num_warps=4)
    return uni, mem, cnt, T


def qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
                                     compress_ratio: int = 4):
    rows = q.shape[0]
    token_topk = (logical_indices.shape[1] - (compress_ratio - 1))
    entries = _qsa_union_entries(logical_indices, compress_ratio, token_topk)
    u4 = _qsa_union_build(entries, 4)
    u2 = _qsa_union_build(entries, 2)
    t4 = float(u4[2].sum()) * _QSA_UNION_C4
    t2 = float(u2[2].sum()) * _QSA_UNION_C2
    uni, mem, cnt, T = u4 if t4 <= t2 else u2
    R = 4 if t4 <= t2 else 2
    group_size = q.shape[1] // k_cache.shape[2]
    request = int(token_to_req[0])
    _qsa_union_attn_kernel[(T, k_cache.shape[2])](
        q, k_cache, v_cache, uni, mem, cnt, block_table, out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), uni.stride(0), mem.stride(0), mem.stride(1),
        out.stride(0), out.stride(1), rows, request, k_cache.shape[0],
        R=R, GP=triton.next_power_of_2(group_size), GROUP_SIZE=group_size, HEAD_DIM=q.shape[2],
        BNB=16, CR=compress_ratio, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        STRIDE_TABLE_REQ=block_table.stride(0), num_warps=8, num_stages=1)
    return out


def _qsa_union_eligible(q, logical_indices, token_to_req) -> bool:
    if not _QSA_UNION or q.shape[0] < _QSA_UNION_MIN_ROWS:
        return False
    # single-request batches only (tiles must not straddle requests); token_to_req is sorted in prefill
    return bool(token_to_req[0] == token_to_req[-1])
'''
GATE_ANCHOR = '''    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out
'''
GATE_NEW = GATE_ANCHOR + '''    if _qsa_union_eligible(q, logical_indices, token_to_req):  # QSA UNION gate
        return qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if MARK not in s: print("  qsaunion not installed"); raise SystemExit
    s = s.replace(GATE_NEW, GATE_ANCHOR); s = s[: s.index(MARK)].rstrip("\n") + "\n"
    open(TARGET, "w").write(s); print("  qsaunion REMOVED")
else:
    if MARK in s: print("  qsaunion already installed"); raise SystemExit
    assert s.count(GATE_ANCHOR) == 1, "gate anchor"
    s = s.replace(GATE_ANCHOR, GATE_NEW) + "\n\n" + ADD
    open(TARGET, "w").write(s); print("  qsaunion INSTALLED in", TARGET)
