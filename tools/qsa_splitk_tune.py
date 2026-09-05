#!/usr/bin/env python3
"""Retune the stock QSA split-K attention kernel's config table on this GPU.

`_select_config(num_rows, num_kv_heads, use_prefill_config, num_columns)` in
vllm/models/qwen4_exp/nvidia/ops/qsa.py returns (BLOCK_N, num_warps, num_tiles,
num_splits) and was tuned on GB300. This sweeps BLOCK_N x num_warps x target
splits per dispatch region on real prefill selection dumps (qsadump2) and on
synthetic decode/verify shapes, timing the public wrapper with the module's
selector monkeypatched, and prints the best config per region next to stock.

Usage: qsa_splitk_tune.py <sel_*.pt> [...]    (run in the serving venv, GPU idle)
"""
import statistics
import sys

import torch
import vllm  # noqa: F401
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
from vllm.triton_utils import triton

DEV = "cuda"
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
WIDTH = TOPK + CR - 1
STOCK = Q._select_config


def timeit(fn, n=5, reps=5):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1000 / n)
    return statistics.median(out)


def forced(block_n, warps, target_splits):
    def select(num_rows, num_kv_heads, use_prefill_config, num_columns):
        num_tiles = triton.cdiv(num_columns, block_n)
        return block_n, warps, num_tiles, min(target_splits, num_tiles)

    return select


def make_prefill(f):
    d = torch.load(f)
    blocks = d["blocks"].to(DEV).to(torch.int32).contiguous()
    vis = d["visible_blocks"].to(DEV).to(torch.int32)
    rows = blocks.shape[0]
    qpos = (vis * CR + 2).to(torch.int64)
    logical = torch.empty((rows, WIDTH + 1), device=DEV, dtype=torch.int32)
    expand_qsa_block_indices(blocks, qpos, vis, CR, TOPK, logical)
    ctx = int(qpos.max().item()) + 1
    npages = (ctx + PAGE - 1) // PAGE
    return rows, logical, npages, 1


def make_decode(rows, ctx, num_requests):
    """Uniform decode/verify batch: each request contributes rows // num_requests rows."""
    blocks_total = ctx // CR
    sel = torch.stack(
        [torch.randperm(blocks_total, device=DEV)[: TOPK // CR] for _ in range(rows)]
    ).to(torch.int32)
    vis = torch.full((rows,), blocks_total, device=DEV, dtype=torch.int32)
    qpos = torch.full((rows,), ctx - 1, device=DEV, dtype=torch.int64)
    logical = torch.empty((rows, WIDTH + 1), device=DEV, dtype=torch.int32)
    expand_qsa_block_indices(sel, qpos, vis, CR, TOPK, logical)
    npages = (ctx + PAGE - 1) // PAGE
    return rows, logical, npages, num_requests


def bench(case, use_prefill_config, configs):
    rows, logical, npages, num_requests = case
    q = (torch.randn(rows, HQ, D, device=DEV) * 0.2).to(torch.bfloat16)
    kv = (torch.randn(npages * num_requests, HKV, PAGE, 2 * D, device=DEV) * 0.2).to(
        torch.bfloat16
    )
    k, v = kv.transpose(1, 2).split(D, dim=-1)
    bt = (
        torch.randperm(npages * num_requests, device=DEV)
        .view(num_requests, npages)
        .to(torch.int32)
    )
    per_req = rows // num_requests
    t2r = (torch.arange(rows, device=DEV) // per_req).to(torch.int32)
    out = torch.empty_like(q)

    def run():
        Q.qsa_sparse_paged_attention(q, k, v, logical, bt, t2r, use_prefill_config, out)

    Q._select_config = STOCK
    stock_cfg = STOCK(rows, HKV, use_prefill_config, WIDTH)
    t_stock = timeit(run)
    ref = out.clone()
    results = []
    for block_n, warps, splits in configs:
        Q._select_config = forced(block_n, warps, splits)
        try:
            t = timeit(run)
            diff = float((out.float() - ref.float()).abs().max())
            results.append((t, block_n, warps, splits, diff))
        except Exception as exc:  # smem / compile limits
            results.append((float("inf"), block_n, warps, splits, str(exc).splitlines()[0][:60]))
    Q._select_config = STOCK
    results.sort(key=lambda r: r[0])
    return stock_cfg, t_stock, results


def report(name, stock_cfg, t_stock, results):
    best = results[0]
    print(
        f"{name}: stock BN={stock_cfg[0]} w={stock_cfg[1]} splits={stock_cfg[3]} "
        f"{t_stock:8.1f} us | best BN={best[1]} w={best[2]} splits={best[3]} "
        f"{best[0]:8.1f} us (x{t_stock / best[0]:.2f}) max|diff|={best[4] if isinstance(best[4], float) else 'n/a'}",
        flush=True,
    )
    for t, bn, w, sp, d in results[:6]:
        print(f"    BN={bn:3d} w={w} splits={sp:2d}: {t:8.1f} us  {d if isinstance(d, str) else ''}", flush=True)


PREFILL_CONFIGS = [
    (bn, w, 1) for bn in (16, 32, 64, 128) for w in (1, 2, 4, 8)
]
DECODE_CONFIGS = [
    (bn, w, sp)
    for bn in (16, 32, 64)
    for w in (1, 2, 4)
    for sp in (1, 2, 4, 8, 16, 32, 64)
]

for f in sys.argv[1:]:
    report(f"prefill {f.split('/')[-1]}", *bench(make_prefill(f), True, PREFILL_CONFIGS))
for rows, ctx, nreq in ((1, 8192, 1), (4, 8192, 1), (4, 8192, 4), (16, 8192, 4), (32, 32768, 8), (64, 32768, 16), (128, 32768, 32), (512, 32768, 128)):
    torch.manual_seed(rows)
    report(f"decode rows={rows} ctx={ctx} reqs={nreq}", *bench(make_decode(rows, ctx, nreq), False, DECODE_CONFIGS))
