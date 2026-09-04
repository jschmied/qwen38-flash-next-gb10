#!/usr/bin/env python3
"""Correctness + timing of the integrated union path (qsa_union_patch.py) against the stock kernel, on a real
selection dump run through vLLM's own expand_qsa_block_indices (so the causal tail tokens are exercised).
Usage: qsa_union_test.py <sel_*.pt>  (v3: single union by context, exact-width sort)   (run with the patched venv; VLLM_QSA_UNION may be unset — calls the union
function directly)."""
import sys, statistics, torch
import vllm  # noqa
from vllm.models.qwen4_exp.nvidia.ops import qsa as Q
from vllm.models.qwen4_exp.nvidia.ops.qsa_indexer import expand_qsa_block_indices
dev="cuda"; torch.manual_seed(0); f=sys.argv[1]
HQ, HKV, D, PAGE, CR, TOPK = 24, 2, 256, 1600, 4, 2048
d=torch.load(f); blocks=d["blocks"].to(dev).to(torch.int32).contiguous(); vis=d["visible_blocks"].to(dev).to(torch.int32)
rows=blocks.shape[0]
# query positions: last-but-one token of the open group -> a 3-token causal tail per row
qpos=(vis*CR + 2).to(torch.int32)
# the real expansion wants block indices [rows, 512] (any order, -1 padded) -> tokens [rows, 2051]
logical=torch.empty((rows, TOPK + CR - 1), device=dev, dtype=torch.int32)
expand_qsa_block_indices(blocks, qpos, vis, CR, TOPK, logical)
ctx=int(qpos.max().item())+1; npages=(ctx+PAGE-1)//PAGE
q=(torch.randn(rows, HQ, D, device=dev)*0.2).to(torch.bfloat16)
kc=(torch.randn(npages, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16); vc=(torch.randn(npages, PAGE, HKV, D, device=dev)*0.2).to(torch.bfloat16)
bt=torch.arange(npages, device=dev, dtype=torch.int32)[None, :].contiguous(); t2r=torch.zeros(rows, device=dev, dtype=torch.int32)
def timeit(fn, n=5, reps=5):
    fn(); torch.cuda.synchronize(); o=[]
    for _ in range(reps):
        s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); o.append(s.elapsed_time(e)*1000/n)
    return statistics.median(o)
Q._QSA_UNION=False
out_ref=torch.empty_like(q); Q.qsa_sparse_paged_attention(q, kc, vc, logical, bt, t2r, out_ref); torch.cuda.synchronize()
t_ref=timeit(lambda: Q.qsa_sparse_paged_attention(q, kc, vc, logical, bt, t2r, out_ref))
out=torch.empty_like(q); Q.qsa_sparse_paged_attention_union(q, kc, vc, logical, bt, t2r, out); torch.cuda.synchronize()
t_u=timeit(lambda: Q.qsa_sparse_paged_attention_union(q, kc, vc, logical, bt, t2r, out))
ent=Q._qsa_union_entries(logical, CR, TOPK); ntail=int(((ent>=0)&(ent%2==1)).sum()); nblk=int(((ent>=0)&(ent%2==0)).sum())
diff=(out.float()-out_ref.float()).abs().max(); tails_ok = ntail == int(((logical>=0).sum(1) - (blocks>=0).sum(1)*CR).sum())
print(f"{f.split('/')[-1]}: rows={rows} ctx={ctx} entries: blocks={nblk} tails={ntail} (tail count consistent: {tails_ok})", flush=True)
print(f"  union path {t_u:8.1f} us (incl. both precomputes + R choice) vs stock {t_ref:8.1f} us -> x{t_ref/t_u:4.2f}   max|diff|={float(diff):.4f}", flush=True)
# gate check: eligible with env on, single request
Q._QSA_UNION=True; print("  eligible:", Q._qsa_union_eligible(q, logical, t2r), flush=True)
