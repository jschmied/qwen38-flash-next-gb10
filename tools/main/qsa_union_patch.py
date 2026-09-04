"""Tile-union QSA sparse attention for prefill (findings 92/96/103/104), v4 — reworked to the review of 2026-09-04:
block-only union (exactly R*block_topk sort inputs: 2048 / 1024), the <= CR-1 causal-tail tokens per row handled by a
separate 16-column pass inside the same online softmax, compress_ratio / token_topk / max_seq_len / num_requests
threaded from the QSA owner (no device->host reads), stock request validation preserved. Env-gated:
VLLM_QSA_UNION=1 (default off), VLLM_QSA_UNION_MIN_ROWS (1024), VLLM_QSA_UNION_R4_MAX_CTX (8192, finding 104).
Targets: ops/qsa.py (kernels + wrapper + gate) and qsa.py (forward_qsa passes the metadata) of the running
interpreter (VLLM_QSA_OPS_PY / VLLM_QSA_PY override). `off` removes both."""
import os, sys
def _targets():
    ops, owner = os.environ.get("VLLM_QSA_OPS_PY"), os.environ.get("VLLM_QSA_PY")
    if not (ops and owner):
        import vllm
        base = os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia")
        ops = ops or os.path.join(base, "ops/qsa.py"); owner = owner or os.path.join(base, "qsa.py")
    return ops, owner
OPS, OWNER = _targets()
MARK = "# ---- QSA UNION (jschmied 2026-09-04) ----"
ADD = MARK + '''
import os as _os
import torch
import triton
import triton.language as tl

_QSA_UNION = _os.environ.get("VLLM_QSA_UNION", "0") not in ("0", "", "false", "False")
_QSA_UNION_MIN_ROWS = int(_os.environ.get("VLLM_QSA_UNION_MIN_ROWS", "1024"))
_QSA_UNION_R4_MAX_CTX = int(_os.environ.get("VLLM_QSA_UNION_R4_MAX_CTX", "8192"))  # finding 104: R=4 up to ~8k ctx
_QSA_UNION_TAIL_COLS = 16
if _QSA_UNION:
    print("QSAUNION active", flush=True)


@triton.jit
def _qsa_union_build_kernel(sorted_ptr, uni_ptr, mem_ptr, cnt_ptr, stride_sorted, stride_uni,
                            stride_mem_t, stride_mem_r, N: tl.constexpr):
    # sorted_ptr[t]: ascending packed (block*8 + row_in_tile), BIG*8+7 for padding; exactly N = R*block_topk
    t = tl.program_id(0)
    i = tl.arange(0, N)
    packed = tl.load(sorted_ptr + t * stride_sorted + i)
    prev = tl.load(sorted_ptr + t * stride_sorted + i - 1, mask=i > 0, other=-8)
    blk = packed // 8
    r = packed % 8
    valid = blk < (1 << 27)
    first = (blk != prev // 8) & valid
    pos = tl.cumsum(first.to(tl.int32)) - 1
    tl.store(uni_ptr + t * stride_uni + pos, blk, mask=first)
    tl.store(mem_ptr + t * stride_mem_t + r * stride_mem_r + pos, tl.full((N,), 1, tl.int8), mask=valid)
    tl.store(cnt_ptr + t, tl.sum(first.to(tl.int32)))


@triton.jit
def _qsa_union_attn_kernel(q_ptr, k_cache_ptr, v_cache_ptr, uni_ptr, mem_ptr, cnt_ptr, tail_ptr, block_table_ptr,
                           token_to_req_ptr, out_ptr,
                           stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
                           stride_v_block, stride_v_token, stride_v_head, stride_uni, stride_mem_t, stride_mem_r,
                           stride_tail, stride_table_req, stride_out_row, stride_out_head,
                           num_rows, num_requests, num_cache_blocks,
                           R: tl.constexpr, GP: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr,
                           BNB: tl.constexpr, CR: tl.constexpr, TAIL_COLS: tl.constexpr, PAGE_SIZE: tl.constexpr,
                           PAGE_TABLE_WIDTH: tl.constexpr):
    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    M: tl.constexpr = R * GP
    BN: tl.constexpr = BNB * CR
    TAIL_PER_ROW: tl.constexpr = CR - 1
    m_off = tl.arange(0, M)
    r_of_m = m_off // GP
    h_of_m = m_off % GP
    dim_offsets = tl.arange(0, HEAD_DIM)
    b_off = tl.arange(0, BNB)
    j_off = tl.arange(0, CR)
    row = tile * R + r_of_m
    # stock contract: rows with an invalid request are masked, the block-table row is clamped
    request = tl.load(token_to_req_ptr + tl.minimum(row, num_rows - 1), mask=row < num_rows, other=-1)
    req_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    # the tile's block-table row: any valid row of the tile (the owner guarantees one request per batch; rows with
    # an invalid id, e.g. padding, are masked above and must not pick the table row)
    tile_request = tl.minimum(tl.max(tl.where(req_ok, request, 0), axis=0), num_requests - 1)
    qmask = (row < num_rows) & (h_of_m < GROUP_SIZE) & req_ok
    first_head = kv_head * GROUP_SIZE
    query = tl.load(q_ptr + row[:, None] * stride_q_row + (first_head + h_of_m[:, None]) * stride_q_head
                    + dim_offsets[None, :], mask=qmask[:, None], other=0.0)
    max_value = tl.full((M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((M,), dtype=tl.float32)
    accumulator = tl.zeros((M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
    # ---- pass 1: the tile's union of whole compressed blocks (shared gather, per-row membership) ----
    ubound = tl.load(cnt_ptr + tile)
    for t in range(0, ubound, BNB):
        emask = (t + b_off) < ubound
        blk = tl.load(uni_ptr + tile * stride_uni + t + b_off, mask=emask, other=-1)
        tok2 = tl.where((blk >= 0)[:, None], blk[:, None] * CR + j_off[None, :], -1)
        logical_token = tl.reshape(tok2, (BN,))
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(block_table_ptr + tile_request * stride_table_req
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
        active = (memt > 0) & valid[None, :] & req_ok[:, None]
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(active, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max
    # ---- pass 2: each row's causal tail (<= CR-1 tokens of its open block), one 16-column tile per tile ----
    tt = tl.arange(0, TAIL_COLS)
    slot_row = tt // TAIL_PER_ROW
    tail_tok = tl.load(tail_ptr + tile * stride_tail + tt, mask=tt < R * TAIL_PER_ROW, other=-1)
    safe_token = tl.maximum(tail_tok, 0)
    logical_page = safe_token // PAGE_SIZE
    page_offset = safe_token % PAGE_SIZE
    valid = (tail_tok >= 0) & (logical_page < PAGE_TABLE_WIDTH)
    physical_page = tl.load(block_table_ptr + tile_request * stride_table_req
                            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1), mask=valid, other=-1)
    valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
    safe_page = tl.maximum(physical_page, 0).to(tl.int64)
    keys = tl.load(k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
                   + kv_head * stride_k_head + dim_offsets[:, None], mask=valid[None, :], other=0.0)
    values = tl.load(v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
                     + kv_head * stride_v_head + dim_offsets[None, :], mask=valid[:, None], other=0.0)
    active = (r_of_m[:, None] == slot_row[None, :]) & valid[None, :] & req_ok[:, None]
    scores = tl.dot(query, keys) * softmax_scale_log2
    scores = tl.where(active, scores, -1.0e20)
    next_max = tl.maximum(max_value, tl.max(scores, axis=1))
    alpha = tl.math.exp2(max_value - next_max)
    probabilities = tl.where(active, tl.math.exp2(scores - next_max[:, None]), 0.0)
    accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
    normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
    has_values = normalizer > 0
    out = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    # stock contract: rows with an invalid request are written as zeros (their query was masked to 0 above)
    smask = (row < num_rows) & (h_of_m < GROUP_SIZE)
    tl.store(out_ptr + row[:, None] * stride_out_row + (first_head + h_of_m[:, None]) * stride_out_head
             + dim_offsets[None, :], out.to(tl.bfloat16), mask=smask[:, None])


def _qsa_union_split(logical_indices: torch.Tensor, compress_ratio: int, token_topk: int):
    """Whole compressed blocks [rows, block_topk] (-1 padded) and the causal tail tokens [rows, CR-1]
    (-1 padded), from the expanded selection (layout of expand_qsa_block_indices: complete blocks first as CR
    consecutive tokens each, then the tail of the open block, then -1; newer builds append a count column)."""
    rows, width = logical_indices.shape
    block_topk = token_topk // compress_ratio
    selection_width = token_topk + compress_ratio - 1
    if width not in (selection_width, selection_width + 1):
        raise ValueError(f"QSA union: unexpected selection width {width} for token_topk {token_topk}, ratio {compress_ratio}")
    cols = logical_indices[:, : block_topk * compress_ratio].reshape(rows, block_topk, compress_ratio)
    first = cols[:, :, 0]
    whole = (first >= 0) & ((first % compress_ratio) == 0) & (cols[:, :, compress_ratio - 1] == first + compress_ratio - 1)
    blocks = torch.where(whole, first // compress_ratio, torch.full_like(first, -1)).to(torch.int32)
    c = whole.sum(dim=1, keepdim=True)
    tcols = (c * compress_ratio + torch.arange(compress_ratio - 1, device=logical_indices.device)[None, :]).clamp(max=selection_width - 1)
    tail = torch.gather(logical_indices, 1, tcols)
    tail = torch.where(tail >= 0, tail, torch.full_like(tail, -1)).to(torch.int32)
    return blocks, tail


def _qsa_union_build(blocks: torch.Tensor, tail: torch.Tensor, R: int):
    rows, E = blocks.shape
    T = (rows + R - 1) // R
    N = R * E                                   # 4*512 = 2048 or 2*512 = 1024: exact powers of two
    assert N & (N - 1) == 0, "block_topk * R must be a power of two"
    dev = blocks.device
    b, tl_ = blocks, tail
    if T * R != rows:
        pad = T * R - rows
        b = torch.cat([b, torch.full((pad, E), -1, device=dev, dtype=torch.int32)])
        tl_ = torch.cat([tl_, torch.full((pad, tail.shape[1]), -1, device=dev, dtype=torch.int32)])
    b = b.view(T, R, E)
    rr = torch.arange(R, device=dev, dtype=torch.int32)[None, :, None]
    packed = torch.where(b >= 0, b * 8 + rr, torch.full_like(b, (1 << 27) * 8 + 7)).view(T, N)
    packed, _ = torch.sort(packed, dim=1)
    uni = torch.full((T, N), -1, device=dev, dtype=torch.int32)
    mem = torch.zeros((T, R, N), device=dev, dtype=torch.int8)
    cnt = torch.empty(T, device=dev, dtype=torch.int32)
    _qsa_union_build_kernel[(T,)](packed, uni, mem, cnt, packed.stride(0), uni.stride(0), mem.stride(0), mem.stride(1),
                                  N=N, num_warps=4)
    tails = torch.full((T, _QSA_UNION_TAIL_COLS), -1, device=dev, dtype=torch.int32)
    tails[:, : R * tail.shape[1]] = tl_.view(T, R * tail.shape[1])
    return uni, mem, cnt, tails, T


def qsa_union_pick_r(max_seq_len: int) -> int:
    """Finding 104: R=4 wins up to ~8k of visible context, R=2 beyond (cost model c4=12.8, c2=9.1 ns/unit)."""
    return 4 if max_seq_len <= _QSA_UNION_R4_MAX_CTX else 2


def qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out, *,
                                     compress_ratio: int, token_topk: int, R: int, num_requests: int):
    rows = q.shape[0]
    blocks, tail = _qsa_union_split(logical_indices, compress_ratio, token_topk)
    uni, mem, cnt, tails, T = _qsa_union_build(blocks, tail, R)
    group_size = q.shape[1] // k_cache.shape[2]
    _qsa_union_attn_kernel[(T, k_cache.shape[2])](
        q, k_cache, v_cache, uni, mem, cnt, tails, block_table, token_to_req, out,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), uni.stride(0), mem.stride(0), mem.stride(1),
        tails.stride(0), block_table.stride(0), out.stride(0), out.stride(1), rows, num_requests, k_cache.shape[0],
        R=R, GP=triton.next_power_of_2(group_size), GROUP_SIZE=group_size, HEAD_DIM=q.shape[2],
        BNB=16, CR=compress_ratio, TAIL_COLS=_QSA_UNION_TAIL_COLS, PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1], num_warps=8, num_stages=1)
    return out


def qsa_union_eligible(num_rows: int, num_requests: int, compress_ratio: int, token_topk: int) -> bool:
    """Decided from CPU metadata only (no device reads): enabled, a prefill-sized single-request batch
    (tiles must not straddle requests), and a block_topk that keeps the sort width a power of two."""
    if not _QSA_UNION or num_rows < _QSA_UNION_MIN_ROWS or num_requests != 1:
        return False
    block_topk = token_topk // compress_ratio
    return compress_ratio & (compress_ratio - 1) == 0 and block_topk & (block_topk - 1) == 0
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
GATE_NEW = GATE_ANCHOR + '''    if union is not None:  # QSA UNION gate: the owner decided eligibility from its CPU metadata
        return qsa_sparse_paged_attention_union(q, k_cache, v_cache, logical_indices, block_table, token_to_req, out,
                                                compress_ratio=union["compress_ratio"], token_topk=union["token_topk"],
                                                R=union["R"], num_requests=union["num_requests"])
'''
SIG_ANCHOR = '''    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches."""'''
SIG_NEW = '''    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    union: dict | None = None,  # QSA UNION: {compress_ratio, token_topk, R, num_requests} or None
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches."""'''
OWNER_ANCHOR = '''        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
        )'''
OWNER_NEW = '''        from .ops.qsa import qsa_sparse_paged_attention, qsa_union_eligible, qsa_union_pick_r  # QSA UNION

        union = None
        indexer = getattr(layer, "indexer", None)
        if indexer is not None:
            num_requests = int(attn_metadata.seq_lens.shape[0])
            if qsa_union_eligible(num_tokens, num_requests, indexer.compress_ratio, indexer.token_topk):
                union = {"compress_ratio": indexer.compress_ratio, "token_topk": indexer.token_topk,
                         "R": qsa_union_pick_r(int(attn_metadata.max_seq_len)), "num_requests": num_requests}
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            union=union,
        )'''
ops = open(OPS).read(); owner = open(OWNER).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if MARK not in ops: print("  qsaunion not installed"); raise SystemExit
    ops = ops.replace(GATE_NEW, GATE_ANCHOR).replace(SIG_NEW, SIG_ANCHOR); ops = ops[: ops.index(MARK)].rstrip("\n") + "\n"
    owner = owner.replace(OWNER_NEW, OWNER_ANCHOR)
    open(OPS, "w").write(ops); open(OWNER, "w").write(owner); print("  qsaunion REMOVED")
else:
    if MARK in ops: print("  qsaunion already installed"); raise SystemExit
    assert ops.count(GATE_ANCHOR) == 1, "gate anchor"; assert ops.count(SIG_ANCHOR) == 1, "signature anchor"; assert owner.count(OWNER_ANCHOR) == 1, "owner anchor"
    ops = ops.replace(SIG_ANCHOR, SIG_NEW).replace(GATE_ANCHOR, GATE_NEW) + "\n\n" + ADD
    owner = owner.replace(OWNER_ANCHOR, OWNER_NEW)
    open(OPS, "w").write(ops); open(OWNER, "w").write(owner); print("  qsaunion INSTALLED in", OPS, "and", OWNER)
