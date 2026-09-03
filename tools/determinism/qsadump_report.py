"""Overlap statistics of QSA block selections (from qsadump_patch dumps): how much of the per-query gather a
tile-union kernel would save. Usage: qsadump_report.py <dir>"""
import sys, glob, torch
for f in sorted(glob.glob(sys.argv[1]+"/sel_*.pt")):
    d=torch.load(f); B=d["blocks"]; vis=d["visible_blocks"]; rows,k=B.shape
    if rows < 8: continue
    sets=[set(x[x>=0].tolist()) for x in B]
    ksel=[len(s) for s in sets]; kmean=sum(ksel)/rows
    # consecutive-row Jaccard
    jac=[len(sets[i]&sets[i+1])/max(1,len(sets[i]|sets[i+1])) for i in range(rows-1)]
    line=f"{f.split('/')[-1]}: rows={rows} k={k} mean|sel|={kmean:.0f} visible[max]={int(vis.max())}  consecutive Jaccard mean={sum(jac)/len(jac):.3f}"
    for T in (16, 64, 128):
        unions=[len(set().union(*sets[i:i+T])) for i in range(0,rows,T)]
        per_row_gather=sum(ksel); tile_gather=sum(unions)
        line+=f" | tile {T}: union/tile={sum(unions)/len(unions):.0f} (vs {T}x{kmean:.0f}) -> gather saved {100*(1-tile_gather/per_row_gather):.0f}%"
    print(line, flush=True)
