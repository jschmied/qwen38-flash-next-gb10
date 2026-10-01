#!/usr/bin/env python3
"""TF #136: an expected-throughput depth controller (mrpmorris) vs TensorFold's confidence gate vs an oracle, on the
K=7 draft log (speed-of-light §5ah), priced with the measured cycle(d) table. Calibration of the drafter's p1 to real
acceptance is fitted on one half of the steps and scored on the other (both ways).
Controller: after j drafts, for every final depth d in [j, kmax] estimate E[tokens](d) = 1 + sum_i prod_{m<=i} q_m with
q_m = calib(p1[m]) for drafted positions and the calibrated per-position conditional acceptance for later ones; stop
when d = j maximises E[tokens](d) / cycle(d). Oracle: d = accepted chain length (>= 1).
Usage: replay_et.py <draftlog.jsonl> <segments.json> <cycle.json>"""
import json, sys

recs = [json.loads(l) for l in open(sys.argv[1])]
segs = json.load(open(sys.argv[2]))["segments"]
cyc_all = json.load(open(sys.argv[3]))
KMAX = 7
POS_DEC = "--pos-dec" in sys.argv          # calibrate p1 by (position, decile) instead of decile alone
BINS = [i / 10 for i in range(11)]


def chain(r):
    k = 0
    while k < len(r["top2"]) and r["fed"][k + 1] == r["tgt"][k]:
        k += 1
    return k


def fit(rows):
    """P(draft m accepted | drafts < m accepted) by position, and by p1 decile (pooled over positions)."""
    reach = [0] * KMAX; ok = [0] * KMAX; b_n = [0] * 10; b_ok = [0] * 10
    for r in rows:
        k = chain(r)
        for m in range(KMAX):
            if k < m:
                break
            reach[m] += 1; ok[m] += k > m
            bi = min(9, int(r["top2"][m][1][0] * 10)); b_n[bi] += 1; b_ok[bi] += k > m
    pos = [ok[m] / reach[m] if reach[m] else 0.0 for m in range(KMAX)]
    dec = [(b_ok[i] + 1) / (b_n[i] + 2) for i in range(10)]           # Laplace-smoothed
    if not POS_DEC:
        return pos, dec
    # per (position, decile), shrunk toward the pooled decile with a prior weight of 20 steps
    n2 = [[0] * 10 for _ in range(KMAX)]; o2 = [[0] * 10 for _ in range(KMAX)]
    for r in rows:
        k = chain(r)
        for m in range(KMAX):
            if k < m:
                break
            bi = min(9, int(r["top2"][m][1][0] * 10)); n2[m][bi] += 1; o2[m][bi] += k > m
    pd = [[(o2[m][i] + 20 * dec[i]) / (n2[m][i] + 20) for i in range(10)] for m in range(KMAX)]
    return pos, pd


def controller(p1, pos, dec, cyc):
    q = lambda m, known: ((dec[m] if POS_DEC else dec)[min(9, int(p1[m] * 10))]) if m < known else pos[m]  # noqa: E731
    j = 1
    while j < KMAX:
        best_d, best_v = None, -1.0
        for d in range(j, KMAX + 1):
            e, prod = 1.0, 1.0
            for i in range(d):
                prod *= q(i, j)
                e += prod
            v = e / cyc[d]
            if v > best_v:
                best_d, best_v = d, v
        if best_d == j:
            return j
        j += 1
    return KMAX


def run(kind):
    rows = [rec for s in segs if s["kind"] == kind for rec in recs[s["lines"][0]:s["lines"][1]]]
    cyc = {int(k): v for k, v in cyc_all[kind].items()}
    cyc.setdefault(1, cyc[2] - (cyc[3] - cyc[2]))
    ks = [chain(r) for r in rows]
    p1s = [[p[1][0] for p in r["top2"]] for r in rows]
    n = len(rows)
    rate = lambda ds: sum(min(k, d) + 1 for k, d in zip(ks, ds)) / sum(cyc[d] for d in ds)  # noqa: E731
    base = rate([5] * n)
    gate = lambda tau: [next((j for j in range(1, KMAX) if p[j] < tau), KMAX) for p in p1s]  # noqa: E731
    halves = (list(range(0, n, 2)), list(range(1, n, 2)))
    ctrl = [0] * n
    for fit_ix, use_ix in (halves, halves[::-1]):
        pos, dec = fit([rows[i] for i in fit_ix])
        for i in use_ix:
            ctrl[i] = controller(p1s[i], pos, dec, cyc)
    taus = [round(0.05 * i, 2) for i in range(1, 20)]
    best_tau = max(taus, key=lambda t: rate(gate(t)))
    pct = lambda ds: round(100 * (rate(ds) / base - 1), 2)  # noqa: E731
    out = {"kind": kind, "steps": n, "vs": "fixed K=5",
           "gate_0.7_depth7": pct(gate(0.7)), f"gate_best_tau{best_tau}": pct(gate(best_tau)),
           "controller_depth7": pct(ctrl), "oracle_depth7": pct([max(1, min(k, KMAX)) for k in ks]),
           "drafts_per_step": {"gate0.7": round(sum(gate(0.7)) / n, 2), "controller": round(sum(ctrl) / n, 2),
                               "oracle": round(sum(max(1, min(k, KMAX)) for k in ks) / n, 2)}}
    out["controller_vs_gate0.7"] = round(100 * (rate(ctrl) / rate(gate(0.7)) - 1), 2)
    out["oracle_vs_gate0.7"] = round(100 * (rate([max(1, min(k, KMAX)) for k in ks]) / rate(gate(0.7)) - 1), 2)
    print(json.dumps(out))


for kind in ("code", "prose"):
    run(kind)
