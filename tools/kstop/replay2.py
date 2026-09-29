#!/usr/bin/env python3
"""Replay of a FNDRAFTLOG at K=7 (tools/kstop/HYPOTHESIS.md): depth cap x confidence-stop designs, priced with a
MEASURED cycle(K) table (ms per verify cycle at c=1 with d drafts and 1+d exact rows, from the kcost run).

Per logged step: k = accepted chain length over the 7 logged drafts (fed[j+1] == tgt[j]), p1[j] = the drafter's top-1
probability at draft step j. A stop before draft j (j >= 1, the first draft is always made) when p1[j] < tau.
Designs (tokens per step = min(k, d) + 1):
  fixed      d = Kmax, no stop                                       time cycle(Kmax)
  exact      stop anywhere, a graph for every draft count            time cycle(d)
  padded     graphs only for even row counts (d odd); a stop at even d pads one row      time cycle(d) + resid
  roundup    same graphs, the stop is rounded up to fill the paid row (d -> d+1 if even) time cycle(d')
  two_tau    tau_new when the next draft opens a new graph size, tau_pad when it fills a paid row; padded pricing
resid = cost of a padding row (unmeasured; bracketed by --resid). Steps are treated as independent (a stop does not
change later steps' text), as in speed-of-light §5v.
Usage: replay2.py <draftlog.jsonl> <segments.json> <cycle.json {"code": {"2": ms, ...}, "prose": {...}}> [--resid 0,1.1,3.3]"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("segments"); ap.add_argument("cycle")
ap.add_argument("--resid", default="0,1.1,3.3")
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.log)]
segs = json.load(open(a.segments))["segments"]
cyc_all = json.load(open(a.cycle))
RES = [float(x) for x in a.resid.split(",")]
TAUS = [round(0.05 * i, 2) for i in range(1, 20)]


def chain(r):
    k = 0
    while k < len(r["top2"]) and r["fed"][k + 1] == r["tgt"][k]:
        k += 1
    return k


def stop_at(p1, tau, kmax, tau_pad=None):
    """Drafts made: stop before draft j when p1[j] < threshold (j >= 1). With tau_pad, the threshold for draft j is
    tau_pad when j+1 rows... i.e. when j drafts give an odd row count (1 + j odd -> j even) that pads up anyway."""
    for j in range(1, kmax):
        t = tau if tau_pad is None else (tau_pad if (1 + j) % 2 == 1 else tau)
        if p1[j] < t:
            return j
    return kmax


def analyse(rows, cyc, kind):
    cyc = {int(k): v for k, v in cyc.items()}
    if 1 not in cyc:  # one draft: extrapolate the measured slope
        cyc[1] = cyc[2] - (cyc[3] - cyc[2])
    ks = [chain(r) for r in rows]
    p1s = [[p[1][0] for p in r["top2"]] for r in rows]
    n = len(rows)
    base = (sum(min(k, 5) + 1 for k in ks) / n) / cyc[5]  # tokens per ms, prod K=5
    out = {"kind": kind, "steps": n, "base_K5_tok_per_s": round(1000 * base, 2),
           "per_pos_accept": [round(sum(k > j for k in ks) / n, 3) for j in range(7)]}

    def rate(tok, ms):
        return round(100 * ((tok / ms) / base - 1), 2)

    for kmax in (5, 6, 7):
        res = {"fixed": rate(sum(min(k, kmax) + 1 for k in ks), n * cyc[kmax])}
        best = {}
        for tau in TAUS:
            ds = [stop_at(p, tau, kmax) for p in p1s]
            tok = sum(min(k, d) + 1 for k, d in zip(ks, ds))
            best.setdefault("exact", []).append((rate(tok, sum(cyc[d] for d in ds)), tau, round(sum(ds) / n, 2)))
            for rs in RES:
                ms_pad = sum(cyc[d] + (rs if (1 + d) % 2 == 1 else 0) for d in ds)
                best.setdefault(f"padded@{rs}", []).append((rate(tok, ms_pad), tau))
                du = [d + 1 if (1 + d) % 2 == 1 and d < kmax else d for d in ds]
                tok_u = sum(min(k, d) + 1 for k, d in zip(ks, du))
                best.setdefault(f"roundup@{rs}", []).append((rate(tok_u, sum(cyc[d] for d in du)), tau))
                for tp in TAUS:
                    d2 = [stop_at(p, tau, kmax, tp) for p in p1s]
                    tok2 = sum(min(k, d) + 1 for k, d in zip(ks, d2))
                    ms2 = sum(cyc[d] + (rs if (1 + d) % 2 == 1 else 0) for d in d2)
                    best.setdefault(f"two_tau@{rs}", []).append((rate(tok2, ms2), tau, tp))
        for key, vals in best.items():
            res[key] = max(vals)
        out[f"Kmax{kmax}"] = res
    return out


for kind in ("code", "prose"):
    rows = [rec for s in segs if s["kind"] == kind for rec in recs[s["lines"][0]:s["lines"][1]]]
    if rows and kind in cyc_all:
        print(json.dumps(analyse(rows, cyc_all[kind], kind)))
