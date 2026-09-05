#!/usr/bin/env python3
"""Stock vs GB10-tuned split-K vs tile-union (whole path) on captured prefill selections, under two cache
conditions: L2-resident (the request's own pages only) and DRAM-bound (pages spread over a 64-page cache).
Usage: qsa_three_way.py <sel_*.pt> [...]   (venv with the tile-union branch; VLLM_QSA_TILE_UNION=1 set)"""
import os
import statistics
import sys

import torch
import vllm  # noqa: F401
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
from vllm.models.qwen4_exp.nvidia.ops import qsa_tile_union as U
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
from vllm.triton_utils import triton

os.environ.setdefault("VLLM_QSA_TILE_UNION", "1")
DEV = "cuda"
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
WIDTH = TOPK + CR - 1
STOCK = Q._select_config
TUNED = {  # finding 120: prefill regions on GB10
    "big": (32, 8, 1),  # bp > 2048, prefill
    "tail": (16, 4, 1),  # bp <= 1024, prefill
}


def timeit(fn, n=5, reps=5):
    fn(); torch.cuda.synchronize(); out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize(); out.append(s.elapsed_time(e) * 1000 / n)
    return statistics.median(out)


def forced(block_n, warps, target_splits):
    def select(num_rows, num_kv_heads, use_prefill_config, num_columns):
        num_tiles = triton.cdiv(num_columns, block_n)
        return block_n, warps, num_tiles, min(target_splits, num_tiles)
    return select


for f in sys.argv[1:]:
    d = torch.load(f)
    blocks = d["blocks"].to(DEV).to(torch.int32).contiguous()
    vis = d["visible_blocks"].to(DEV).to(torch.int32)
    rows = blocks.shape[0]
    qpos = (vis * CR + 2).to(torch.int64)
    logical = torch.empty((rows, WIDTH + 1), device=DEV, dtype=torch.int32)
    expand_qsa_block_indices(blocks, qpos, vis, CR, TOPK, logical)
    ctx = int(qpos.max().item()) + 1
    npages = (ctx + PAGE - 1) // PAGE
    q = (torch.randn(rows, HQ, D, device=DEV) * 0.2).to(torch.bfloat16)
    t2r = torch.zeros(rows, device=DEV, dtype=torch.int32)
    qsl = torch.tensor([0, rows], device=DEV, dtype=torch.int32)
    inputs = U.QSATileUnionInputs(block_indices=blocks, logical_positions=qpos, query_start_loc=qsl,
                                  num_decode_tokens=0, num_prefills=1, compress_ratio=CR, token_topk=TOPK)
    cfg = TUNED["big"] if rows * HKV > 2048 else TUNED["tail"]
    for cache_pages, label in ((npages, "L2-resident"), (64, "DRAM-bound")):
        torch.manual_seed(1)
        kv = (torch.randn(cache_pages, HKV, PAGE, 2 * D, device=DEV) * 0.2).to(torch.bfloat16)
        k, v = kv.transpose(1, 2).split(D, dim=-1)
        bt = torch.randperm(cache_pages, device=DEV)[:npages].view(1, npages).to(torch.int32).contiguous()
        out = torch.empty_like(q)
        Q._select_config = STOCK
        t_stock = timeit(lambda: Q.qsa_sparse_paged_attention(q, k, v, logical, bt, t2r, True, out))
        ref = out.clone()
        Q._select_config = forced(*cfg)
        t_tuned = timeit(lambda: Q.qsa_sparse_paged_attention(q, k, v, logical, bt, t2r, True, out))
        d_tuned = float((out.float() - ref.float()).abs().max())
        Q._select_config = STOCK
        t_union = timeit(lambda: Q.qsa_sparse_paged_attention(q, k, v, logical, bt, t2r, True, out, tile_union=inputs))
        d_union = float((out.float() - ref.float()).abs().max())
        print(f"{f.split('/')[-1]} rows={rows} ctx={ctx} [{label}, {cache_pages} pages]: stock {t_stock:8.1f} us | "
              f"tuned(BN{cfg[0]},w{cfg[1]}) {t_tuned:8.1f} us (x{t_stock / t_tuned:.2f}, diff {d_tuned:.0e}) | "
              f"union {t_union:8.1f} us (x{t_stock / t_union:.2f}, diff {d_union:.0e})", flush=True)
