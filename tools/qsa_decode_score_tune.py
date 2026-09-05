#!/usr/bin/env python3
"""Sweep the QSA decode scoring kernel (BLOCK_N, STAGES, num_warps) and compare persistent vs cooperative top-k on GB10.

`qsa_select_paged_decode` hard-codes BLOCK_N=64, STAGES=2, num_warps=2; cooperative_topk is gated off capability 12x.
Synthetic request-major decode batches at the production geometry (4 heads x 128, compressed page 400 rows). Logits are
checked equal to the stock configuration. Usage: qsa_decode_score_tune.py (serving venv)"""
import statistics, itertools, torch, vllm  # noqa
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer as I
from vllm.triton_utils import triton
DEV = "cuda"; H, D, PAGE, CR, TOPK = 4, 128, 400, 4, 2048
torch.manual_seed(0)
def timeit(fn, n=10, reps=5):
    fn(); torch.cuda.synchronize(); out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); out.append(s.elapsed_time(e) * 1000 / n)
    return statistics.median(out)
def tiles(nreq, columns, bn):
    programs = nreq * triton.cdiv(columns, bn)
    return 1 if programs < 16384 else 2 if programs < 32768 else 4 if programs < 131072 else 8
for R, L, ctx in ((1, 4, 8192), (4, 4, 8192), (16, 4, 32768), (1, 1, 8192), (16, 1, 32768), (64, 4, 32768)):
    groups = ctx // CR; pages = triton.cdiv(groups, PAGE); columns = pages * PAGE
    q = (torch.randn(R * L, H, D, device=DEV) * 0.3).to(torch.bfloat16)
    kc = (torch.randn(R * pages + 1, PAGE, 1, D, device=DEV) * 0.3).to(torch.bfloat16)
    pt = (torch.arange(R * pages, device=DEV, dtype=torch.int32) + 1).view(R, pages)
    vis = torch.full((R * L,), groups, device=DEV, dtype=torch.int32)
    def score(bn, st, w):
        logits = torch.empty((R * L, columns), dtype=torch.float32, device=DEV); tp = tiles(R, columns, bn)
        grid = (R, triton.cdiv(columns, bn * tp))
        def run():
            I._qsa_mqa_paged_uniform_kernel[grid](q, kc, pt, vis, logits, *q.stride()[:-1], *kc.stride()[:2], *pt.stride()[:-1], *logits.stride()[:-1],
                PAGE_SIZE=PAGE, PAGE_TABLE_WIDTH=pages, NUM_HEADS=H, HEAD_DIM=D, DECODE_QUERY_LEN=L, BLOCK_N=bn, TILES_PER_PROG=tp, STAGES=st, num_warps=w)
        return run, logits
    run, ref = score(64, 2, 2); t_stock = timeit(run); rows = []
    for bn, st, w in itertools.product((32, 64, 128), (1, 2, 3), (1, 2, 4)):
        try:
            run, lg = score(bn, st, w); t = timeit(run); rows.append((t, bn, st, w, bool(torch.equal(lg, ref))))
        except Exception as ex:
            rows.append((float("inf"), bn, st, w, str(ex).splitlines()[0][:50]))
    rows.sort(key=lambda r: r[0])
    print(f"score R={R} L={L} ctx={ctx}: stock BN=64 st=2 w=2 {t_stock:8.1f} us | best BN={rows[0][1]} st={rows[0][2]} w={rows[0][3]} {rows[0][0]:8.1f} us (x{t_stock/rows[0][0]:.2f}) identical={rows[0][4]}", flush=True)
    for t, bn, st, w, ok in rows[:4]: print(f"    BN={bn} st={st} w={w}: {t:8.1f} us identical={ok}", flush=True)
    # top-k: persistent (GB10 default) vs cooperative (gated off 12x)
    rows_n = R * L; block_topk = TOPK // CR
    if rows_n <= 64:
        out_p = torch.empty(rows_n, block_topk, dtype=torch.int32, device=DEV); out_c = torch.empty_like(out_p)
        ws = torch.empty(64 << 20, dtype=torch.uint8, device=DEV)
        tp_ = timeit(lambda: torch.ops._C.persistent_topk(ref, vis, out_p, ws, block_topk, columns))
        try:
            tc_ = timeit(lambda: torch.ops._C.cooperative_topk(ref, vis, out_c, ws, block_topk, columns)); torch.cuda.synchronize()
            same = torch.equal(torch.sort(out_p, dim=1).values, torch.sort(out_c, dim=1).values)
            print(f"    topk rows={rows_n}: persistent {tp_:8.1f} us | cooperative {tc_:8.1f} us (x{tp_/tc_:.2f}) same-set={same}", flush=True)
        except Exception as ex:
            print(f"    topk rows={rows_n}: persistent {tp_:8.1f} us | cooperative FAILED: {str(ex).splitlines()[0][:80]}", flush=True)
