#!/usr/bin/env python3
"""Offline replay of FNDRAFTLOG records (greedy target): what would a two-candidate branch or a confidence stop have
bought on the logged steps? Record: fed = [last token, d1..dK], tgt[j] = target's greedy token after fed[:j+1],
top2[j] = ([id1, id2], [p1, p2]) of the drafter at draft step j (it drafted d_{j+1}).

Cost model (ms, c=1, K drafts): step = T0 + K*(DRAFT + ROW); an extra verify row adds ROW. Defaults from speed-of-light
(verify row ~3.3 ms from the expert union, NVFP4 draft step ~1.3 ms, K=5 code step ~66 ms). Override with --t0/--draft/--row.
Usage: replay.py <draftlog.jsonl> <segments.json> [--t0 43] [--draft 1.3] [--row 3.3]"""
import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("log"); ap.add_argument("segments")
ap.add_argument("--t0", type=float, default=43.0); ap.add_argument("--draft", type=float, default=1.3)
ap.add_argument("--row", type=float, default=3.3)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.log)]
segs = json.load(open(a.segments))["segments"]


def analyse(rows, name):
    K = len(rows[0]["top2"])
    step = a.t0 + K * (a.draft + a.row)
    n = len(rows); acc = []; brk = []; resc = 0; breaks = 0
    gain = {"branch_pos1": 0, "branch_min_margin": 0}
    cond = {t: [0, 0] for t in (0.1, 0.2, 0.3, 0.5)}           # margin threshold -> [gain tokens, rows spent]
    for r in rows:
        fed, tgt, top2 = r["fed"], r["tgt"], r["top2"]
        k = 0
        while k < K and fed[k + 1] == tgt[k]:
            k += 1
        acc.append(k)
        margins = [p[1][0] - p[1][1] for p in top2]
        m = min(range(K), key=lambda j: margins[j])
        if k < K:
            breaks += 1
            hit = top2[k][0][1] == tgt[k]
            resc += hit
            if k == 0 and hit: gain["branch_pos1"] += 1
            if k == m and hit: gain["branch_min_margin"] += 1
            for t in cond:
                if margins[m] < t and k == m and hit: cond[t][0] += 1
        for t in cond:
            if margins[m] < t: cond[t][1] += 1
    tok = sum(x + 1 for x in acc)
    base_rate = tok / (n * step)                                 # tokens per ms, fixed chain
    out = {"cell": name, "steps": n, "K": K, "tokens_per_step": round(tok / n, 3),
           "per_pos_accept": [round(sum(x > j for x in acc) / n, 3) for j in range(K)],
           "break_rate": round(breaks / n, 3), "top2_rescue_at_break": round(resc / max(1, breaks), 3)}
    for g, v in gain.items():                                    # fixed one extra row every step
        rate = (tok + v) / (n * (step + a.row))
        out[g] = {"extra_tokens_per_step": round(v / n, 3), "speedup_pct": round(100 * (rate / base_rate - 1), 2)}
    out["conditional_min_margin"] = {                            # extra row only when margin < t (variable shapes)
        str(t): {"branch_frac": round(c[1] / n, 3), "extra_tokens_per_step": round(c[0] / n, 3),
                 "speedup_pct": round(100 * (((tok + c[0]) / (n * step + c[1] * a.row)) / base_rate - 1), 2)}
        for t, c in cond.items()}
    stop = {}                                                    # TF1: stop before the first draft with p1 < tau
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5):
        t2 = ms = nd = 0.0
        for r, k in zip(rows, acc):
            p1 = [p[1][0] for p in r["top2"]]
            d = next((j for j in range(1, K) if p1[j] < tau), K)   # always keep the first draft
            t2 += min(k, d) + 1; ms += a.t0 + d * (a.draft + a.row); nd += d
        stop[str(tau)] = {"mean_drafts": round(nd / len(rows), 2), "speedup_pct": round(100 * ((t2 / ms) / base_rate - 1), 2)}
    out["confidence_stop"] = stop
    return out


for kind in ("code", "prose"):
    rows = [rec for s in segs if s["kind"] == kind for rec in recs[s["lines"][0]:s["lines"][1]]]
    if rows:
        print(json.dumps(analyse(rows, kind), indent=1))
