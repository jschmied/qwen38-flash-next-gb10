#!/usr/bin/env python3
"""Byte ledger, prefill window: kernel families by time from the nsys sqlite (prefill = kernels before the first
CUDA-graph node), then per-call grid and median duration for the memory-bound families. Bytes are computed by hand
from the grids (see speed-of-light §5aa). Usage: ledger_prefill.py fn.sqlite"""
import sqlite3, sys, collections
c = sqlite3.connect(sys.argv[1]); S = dict(c.execute("select id,value from StringIds"))
K = list(c.execute("select start,end,shortName,gridX,gridY,gridZ,graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
fg = next(k for k in K if k[6]); pre = [k for k in K if k[0] < fg[0]]
print("prefill window ms %.1f, kernels %d" % ((fg[0] - K[0][0]) / 1e6, len(pre)))
fam = collections.defaultdict(lambda: [0, 0.0])
for k in pre:
    f = fam[S.get(k[2], "?")]; f[0] += 1; f[1] += (k[1] - k[0]) / 1e6
tot = sum(v[1] for v in fam.values())
for n, (cnt, ms) in sorted(fam.items(), key=lambda x: -x[1][1])[:30]:
    print("%7.1f ms %5.1f%% %5d  %s" % (ms, 100 * ms / tot, cnt, n[:100]))
for name in ("_hc_combine_norm_kernel", "_hc_gate_mix_kernel", "finalizeMoeRoutingKernel", "doActivationKernel",
             "expandInputRowsKernel", "cvt_fp16_to_fp4", "per_token_group_quant_8bit_kernel", "layer_norm_fwd_kernel"):
    g = collections.defaultdict(list)
    for k in pre:
        if S.get(k[2]) == name: g[(k[3], k[4], k[5])].append((k[1] - k[0]) / 1e3)
    for grid, v in sorted(g.items(), key=lambda x: -len(x[1]))[:3]:
        v.sort(); print(name, grid, "n", len(v), "median us %.1f" % v[len(v) // 2])
