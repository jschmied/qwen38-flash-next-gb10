#!/usr/bin/env python3
"""Asserting test of the integrated QSA union path (tools/main/qsa_union_patch.py v4) against the stock kernel.

Runs on real selection dumps (qsadump2: `blocks` [rows, block_topk], `visible_blocks` [rows]) pushed through vLLM's
own expand_qsa_block_indices, so the causal tail of the open block is exercised for every tail length 0..CR-1.
Softmax is deliberately peaked (|q| large) so that a leaked or dropped token moves the output by O(0.1), far above
the bf16 order-of-summation noise (~1e-3); a negative control proves the test has that power.

Usage:  qsa_union_test.py <sel_*.pt> [<sel_*.pt> ...]     (patched venv; VLLM_QSA_UNION may be unset)
        pytest tools/qsa_union_test.py                     (QSA_DUMPS=<dir or files>, default results/qsadump2)
Exit status 1 on the first failed assertion."""
import glob, os, statistics, sys
import torch
import vllm  # noqa
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices

DEV = "cuda"
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
TOL = 2e-2          # max |union - stock| on bf16 outputs with peaked softmax (measured noise ~1e-3)
NEG_MIN = 5e-2      # a single swapped block per row must move the output by at least this much (test power)


def _dumps():
    spec = os.environ.get("QSA_DUMPS", "/opt/llm/runners/results/qsadump2")
    files = []
    for s in spec.split(":"):
        files += sorted(glob.glob(os.path.join(s, "sel_*.pt"))) if os.path.isdir(s) else [s]
    return files[:3]


def timeit(fn, n=5, reps=5):
    fn(); torch.cuda.synchronize(); o = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e) * 1000 / n)
    return statistics.median(o)


class Case:
    """One dump at one tail length, with a random physical-page permutation and a 3-row block table."""

    def __init__(self, f, tail_len, seed=0, nreq=3, req=1):
        g = torch.Generator(device=DEV).manual_seed(seed)
        d = torch.load(f)
        self.blocks = d["blocks"].to(DEV).to(torch.int32).contiguous()
        vis = self.vis = d["visible_blocks"].to(DEV).to(torch.int32)
        self.rows = rows = self.blocks.shape[0]
        # query position inside its open block: tail_len tokens (0..CR-1) of that block precede-or-are the query
        self.qpos = (vis * CR + tail_len - 1).clamp(min=0).to(torch.int32) if tail_len else (vis * CR - 1).clamp(min=0).to(torch.int32)
        self.tail_len = tail_len
        self.logical = torch.empty((rows, TOPK + CR - 1), device=DEV, dtype=torch.int32)
        expand_qsa_block_indices(self.blocks, self.qpos, vis, CR, TOPK, self.logical)
        ctx = int(self.qpos.max().item()) + 1
        npages = (ctx + PAGE - 1) // PAGE
        # peaked softmax: scores ~ N(0, 2^2) over 2048 tokens -> a handful of tokens carry the output
        self.q = (torch.randn(rows, HQ, D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
        phys = npages * nreq + 2
        self.kc = torch.randn(phys, PAGE, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
        self.vc = (torch.randn(phys, PAGE, HKV, D, device=DEV, generator=g) * 0.2).to(torch.bfloat16)
        perm = torch.randperm(phys, device=DEV, generator=g)[: npages * nreq].view(nreq, npages).to(torch.int32)
        self.bt = perm.contiguous()                      # rows 0 and 2 are other requests' (decoy) pages
        self.nreq, self.req = nreq, req
        self.t2r = torch.full((rows,), req, device=DEV, dtype=torch.int32)

    def stock(self, logical=None, t2r=None):
        out = torch.empty_like(self.q)
        Q.qsa_sparse_paged_attention(self.q, self.kc, self.vc, self.logical if logical is None else logical,
                                     self.bt, self.t2r if t2r is None else t2r, out)
        return out

    def union(self, R, t2r=None, raw=False):
        out = torch.empty_like(self.q)
        Q.qsa_sparse_paged_attention_union(self.q, self.kc, self.vc, self.logical, self.bt,
                                           self.t2r if t2r is None else t2r, out,
                                           compress_ratio=CR, token_topk=TOPK, R=R, num_requests=self.nreq,
                                           raw=self.raw if raw else None)
        return out

    @property
    def raw(self):
        """What indexer_qsa.py stashes: (block_indices, query positions, visible_blocks, expanded buffer)."""
        return (self.blocks, self.qpos, self.vis, self.logical)


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


def test_split_matches_expansion():
    """_qsa_union_split recovers exactly the whole blocks and the tail tokens the expansion emitted."""
    for f in _dumps():
        for tail_len in range(CR):
            c = Case(f, tail_len)
            blocks, tail = Q._qsa_union_split(c.logical, CR, TOPK)
            n_sel = (c.logical >= 0).sum(1)
            n_blk = (blocks >= 0).sum(1)
            n_tail = (tail >= 0).sum(1)
            assert torch.equal(n_sel, n_blk * CR + n_tail), f"{f}: token count mismatch at tail {tail_len}"
            assert int(n_tail.max()) <= CR - 1
            # every tail token lies inside the query's open block, at or before the query
            tv = tail[tail >= 0]
            rq = c.qpos[:, None].expand_as(tail)[tail >= 0]
            assert bool(((tv <= rq) & (tv // CR == rq // CR)).all()), "tail token outside the open block"
            # the whole blocks are exactly the dump's selection (as a set per row)
            sb, _ = torch.sort(torch.where(blocks >= 0, blocks, torch.full_like(blocks, 1 << 30)), 1)
            db, _ = torch.sort(torch.where(c.blocks >= 0, c.blocks, torch.full_like(c.blocks, 1 << 30)), 1)
            assert torch.equal(sb, db), "whole-block set differs from the dump"
            # the newer main appends a count column: the split must accept width + 1 unchanged
            wide = torch.cat([c.logical, n_sel.to(torch.int32)[:, None]], 1)
            b2, t2 = Q._qsa_union_split(wide, CR, TOPK)
            assert torch.equal(b2, blocks) and torch.equal(t2, tail)


def test_union_matches_stock_all_tails():
    """Union output == stock output (peaked softmax) for tail lengths 0..CR-1, R=4 and R=2, permuted pages."""
    for f in _dumps():
        for tail_len in range(CR):
            c = Case(f, tail_len, seed=tail_len)
            ref = c.stock()
            for R in (4, 2):
                d = maxdiff(c.union(R), ref)
                assert d < TOL, f"{os.path.basename(f)} tail {tail_len} R={R}: max|diff| {d:.4f} >= {TOL}"


def test_raw_build_matches_split_and_stock():
    """Lever 1: the union built from the indexer's selection equals the split-based build and the stock output."""
    for f in _dumps():
        for tail_len in range(CR):
            c = Case(f, tail_len, seed=7 + tail_len)
            ref = c.stock()
            for R in (2, 4):
                blocks, tail = Q._qsa_union_split(c.logical, CR, TOPK)
                a = Q._qsa_union_build(blocks, tail, R, c.t2r, c.bt, c.nreq, c.kc, CR)
                b = Q._qsa_union_build_raw(c.blocks, c.qpos, c.vis, R, c.t2r, c.bt, c.nreq, c.kc, CR)
                for x, y, name in zip(a[:4], b[:4], ("uni", "mem", "cnt", "tails")):
                    assert torch.equal(x, y), f"{os.path.basename(f)} tail {tail_len} R={R}: {name} differs between raw and split"
                d = maxdiff(c.union(R, raw=True), ref)
                assert d < TOL, f"{os.path.basename(f)} tail {tail_len} R={R} raw: max|diff| {d:.4f}"
            # a stale stash (different buffer) must fall back to the split path, not be used
            stale = (c.blocks, c.qpos, c.vis, c.logical.clone())
            out = torch.empty_like(c.q)
            Q.qsa_sparse_paged_attention_union(c.q, c.kc, c.vc, c.logical, c.bt, c.t2r, out, compress_ratio=CR,
                                               token_topk=TOPK, num_requests=c.nreq, raw=stale)
            assert maxdiff(out, ref) < TOL


def test_negative_control_has_power():
    """Swapping one selected block per row for a random other block moves the stock output by >> TOL."""
    f = _dumps()[0]
    c = Case(f, 2)
    ref = c.stock()
    bad = c.logical.clone()
    rows = torch.arange(c.rows, device=DEV)
    # replace the first whole block (columns 0..CR-1) by a block from far away in the context (still < qpos)
    victim = (c.qpos // CR) // 2
    bad[:, :CR] = (victim[:, None] * CR + torch.arange(CR, device=DEV)[None, :]).to(torch.int32)
    d = maxdiff(c.stock(logical=bad), ref)
    assert d > NEG_MIN, f"negative control moved the output by only {d:.4f}: the tolerance {TOL} is not meaningful"


def test_invalid_request_rows_masked():
    """Rows whose request id is invalid produce the same (zero) output as in the stock kernel."""
    c = Case(_dumps()[0], 3)
    t2r = c.t2r.clone()
    t2r[::7] = -1
    t2r[3::11] = c.nreq + 5
    ref = c.stock(t2r=t2r)
    assert float(ref[::7].float().abs().max()) == 0.0
    for R in (4, 2):
        out = c.union(R, t2r=t2r)
        assert float(out[::7].float().abs().max()) == 0.0
        assert float(out[3::11].float().abs().max()) == 0.0
        assert maxdiff(out, ref) < TOL


def test_eligibility_is_cpu_only():
    on = Q._QSA_UNION
    try:
        Q._QSA_UNION = True
        assert Q.qsa_union_eligible(4096, 1, CR, TOPK)
        assert not Q.qsa_union_eligible(Q._QSA_UNION_MIN_ROWS - 1, 1, CR, TOPK)
        assert not Q.qsa_union_eligible(4096, 2, CR, TOPK)          # tiles must not straddle requests
        assert not Q.qsa_union_eligible(4096, 1, CR, 1536)         # block_topk 384: sort width not a power of two
        assert not Q.qsa_union_eligible(4096, 1, 3, 2049)
        assert Q.qsa_union_layout_ok(torch.empty(2, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16),
                                     torch.empty(2, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16), CR)
        assert not Q.qsa_union_layout_ok(torch.empty(2, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16),
                                         torch.empty(2, PAGE, HKV, D, device=DEV, dtype=torch.bfloat16), 3)
        Q._QSA_UNION = False
        assert not Q.qsa_union_eligible(4096, 1, CR, TOPK)
    finally:
        Q._QSA_UNION = on


def report_timing():
    for f in _dumps():
        c = Case(f, 2)
        ctx = int(c.qpos.max().item()) + 1
        t_ref = timeit(c.stock)
        line = f"{os.path.basename(f)}: rows={c.rows} ctx={ctx} stock {t_ref:8.1f} us"
        for R in (2, 4):
            t_u = timeit(lambda: c.union(R))
            t_r = timeit(lambda: c.union(R, raw=True))
            line += f" | R={R} union(split) {t_u:8.1f} us x{t_ref / t_u:4.2f}, union(raw) {t_r:8.1f} us x{t_ref / t_r:4.2f}"
        print(line, flush=True)
        # components: split (torch), build (torch + sort + Triton), kernel
        for R in (2, 4):
            t_split = timeit(lambda: Q._qsa_union_split(c.logical, CR, TOPK))
            blocks, tail = Q._qsa_union_split(c.logical, CR, TOPK)
            t_build = timeit(lambda: Q._qsa_union_build(blocks, tail, R, c.t2r, c.bt, c.nreq, c.kc, CR))
            t_raw = timeit(lambda: Q._qsa_union_build_raw(c.blocks, c.qpos, c.vis, R, c.t2r, c.bt, c.nreq, c.kc, CR))
            print(f"    R={R}: split {t_split:7.1f} + build {t_build:7.1f} us | raw build {t_raw:7.1f} us | kernel ~{timeit(lambda: c.union(R)) - t_split - t_build:7.1f} us", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["QSA_DUMPS"] = ":".join(sys.argv[1:])
    for t in (test_split_matches_expansion, test_negative_control_has_power, test_union_matches_stock_all_tails,
              test_raw_build_matches_split_and_stock, test_invalid_request_rows_masked, test_eligibility_is_cpu_only):
        t(); print(f"  PASS {t.__name__}", flush=True)
    report_timing()
