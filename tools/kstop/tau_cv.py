#!/usr/bin/env python3
"""tau curve, prompt-level cross-validation and two-candidate verification on the K=7 draft log (§5ah follow-up).
Costs: measured cycle(d) (cycle-k.json); one extra verify row = ROW ms (slope 4.8 minus a ~1.3 ms draft step).
A second candidate verified at position j gains exactly one token iff the chain breaks at j and the drafter's second
choice is the target's token there (the target's token at j is emitted anyway; the extra row buys position j+1)."""
import itertools, json, sys
log, segf, cycf = sys.argv[1:4]
ROW = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
recs = [json.loads(l) for l in open(log)]
segs = json.load(open(segf))["segments"]
cyc_all = json.load(open(cycf))
TAUS = [round(0.05 * i, 2) for i in range(6, 20)]


def chain(r):
    k = 0
    while k < 7 and r["fed"][k + 1] == r["tgt"][k]:
        k += 1
    return k


def gain(rows, cyc, tau, kmax=7, branch=None):
    """% vs fixed K=5; branch = p2 threshold for verifying the 2nd candidate at the stop position (None = off)."""
    tok = ms = tok5 = ms5 = 0.0
    for r in rows:
        k = chain(r); p = r["top2"]
        tok5 += min(k, 5) + 1; ms5 += cyc[5]
        d = next((j for j in range(1, kmax) if p[j][1][0] < tau), kmax)
        t = min(k, d) + 1; m = cyc[d]
        if branch is not None and k >= d - 1 and d < kmax + 1:
            # the stop cut at d: the last verified draft is position d-1 (0-based); a break exactly there can be rescued
            j = d - 1
            if p[j][1][1] >= branch:
                m += ROW
                if k == j and p[j][0][1] == r["tgt"][j]:
                    t += 1
        tok += t; ms += m
    return 100 * ((tok / ms) / (tok5 / ms5) - 1)


out = {}
for kind in ("code", "prose"):
    cyc = {int(k): v for k, v in cyc_all[kind].items()}
    cyc[1] = cyc[2] - (cyc[3] - cyc[2])
    by_prompt = [[r for r in recs[s["lines"][0]:s["lines"][1]]] for s in segs if s["kind"] == kind]
    rows = [r for b in by_prompt for r in b]
    curve = {t: round(gain(rows, cyc, t), 2) for t in TAUS}
    cv = []
    for train in itertools.combinations(range(len(by_prompt)), len(by_prompt) // 2):
        tr = [r for i in train for r in by_prompt[i]]
        te = [r for i in range(len(by_prompt)) if i not in train for r in by_prompt[i]]
        best = max(TAUS, key=lambda t: gain(tr, cyc, t))
        cv.append({"train": list(train), "tau_fit": best, "test_gain_at_fit": round(gain(te, cyc, best), 2),
                   "test_gain_at_0.8": round(gain(te, cyc, 0.8), 2),
                   "test_best": round(max(gain(te, cyc, t) for t in TAUS), 2)})
    # breaks and the second candidate
    brk = [(r, chain(r)) for r in rows]
    brk = [(r, k) for r, k in brk if k < 7]
    sec = sum(r["top2"][k][0][1] == r["tgt"][k] for r, k in brk) / len(brk)
    near = [(r, k) for r, k in brk if 0.35 <= r["top2"][k][1][0] <= 0.65 and r["top2"][k][1][1] >= 0.25]
    sec_near = sum(r["top2"][k][0][1] == r["tgt"][k] for r, k in near) / max(1, len(near))
    br = {str(b): round(gain(rows, cyc, 0.8, branch=b), 2) for b in (0.15, 0.25, 0.35, 0.45)}
    out[kind] = {"curve_exact_depth7": curve, "cv_half_splits": cv,
                 "breaks": len(brk), "second_right_at_break": round(sec, 3),
                 "near_5050_breaks": len(near), "second_right_near_5050": round(sec_near, 3),
                 "stop0.8_plus_branch_at_stop_by_p2": br, "stop0.8_no_branch": round(gain(rows, cyc, 0.8), 2)}
print(json.dumps(out, indent=1))
