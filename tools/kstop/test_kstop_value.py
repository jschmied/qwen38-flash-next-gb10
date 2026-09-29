#!/usr/bin/env python3
"""FNKSTOP_AGG=value unit test on the GPU: the batch draft count by expected value, inside the IF-node graph as the patch
captures it, against a float64 Python reference over random batches (1..7 requests, 7 draft steps). Checks graph-body runs
(early exit), d_max (the sizing value) and, for one request, equality with the tau rule. Mismatches that come from a
near-tie in the value comparison (|diff| < 1e-5 relative) are reported separately, not as failures. ONE json; exit 1 on
a real mismatch."""
import json, math, os, random, sys
os.environ["FN_KSTOP"] = "1"
os.environ["FN_KSTOP_AGG"] = "value"
os.environ.setdefault("FN_KSTOP_TAU", "0.75")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import fn_kstop as K

dev = "cuda"
MAXR, V, STEPS, TAU = 8, 64, 7, K.TAU
A, D, R, AL = K._A, K._D, K._R, K._ALPHA
K.set_nreq(1, MAXR, dev)
pred = K.pred(MAXR, dev)
step = torch.zeros((), dtype=torch.int64, device=dev)
logits = torch.zeros(MAXR, V, device=dev)
count = torch.zeros((), dtype=torch.int32, device=dev)


def logits_for(confs):
    out = torch.zeros((len(confs), V))
    for i, c in enumerate(confs):
        c = min(max(c, 1.0 / V + 1e-4), 0.999)
        out[i, 0] = math.log(c * (V - 1) / (1 - c))
    return out


def body():
    count.add_(1)
    K.update(logits, step)


body()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
torch.cuda.synchronize()
with torch.cuda.graph(g, stream=s):
    g.begin_capture_to_if_node(pred)
    body()
    g.end_capture_to_conditional_node()
torch.cuda.synchronize()


def cost(d, n):
    return A + (D + R * n ** AL) * d


def reference(confs, n):
    act = [True] * n; d = [1] * n; surv = [confs[0][r] for r in range(n)]
    cume = sum(surv); best = (n + cume) / cost(1, n); dbest = 1; ties = False
    ub = (n + cume + sum(surv)) / cost(2, n)
    go = any(act) and (n == 1 or ub > best)
    runs = 0
    for j in range(1, STEPS):
        if not go:
            break
        runs += 1
        for r in range(n):
            if act[r] and confs[j][r] >= TAU:
                surv[r] *= confs[j][r]; d[r] = j + 1
            else:
                act[r] = False; surv[r] = 0.0
        cume += sum(surv)
        val = (n + cume) / cost(j + 1, n)
        ties |= abs(val - best) < 1e-5 * best
        if val > best:
            best, dbest = val, j + 1
        ub = (n + cume + sum(surv)) / cost(j + 2, n)
        ties |= abs(ub - best) < 1e-5 * best
        go = any(act) and (n == 1 or ub > best)
    return runs, (max(d) if n == 1 else dbest), ties


rng = random.Random(1)
bad, tie_bad, hist = [], 0, {}
for trial in range(400):
    n = rng.randint(1, MAXR - 1)

    def draw():
        c = rng.random()
        return c if abs(c - TAU) > 0.01 else draw()
    confs = [[draw() for _ in range(MAXR)] for _ in range(STEPS)]
    K.set_nreq(n, MAXR, dev)
    count.zero_()
    step.fill_(0); logits.copy_(logits_for(confs[0]).to(dev)); K.update(logits, step)
    for j in range(1, STEPS):
        step.fill_(j); logits.copy_(logits_for(confs[j]).to(dev))
        g.replay()
    K.copy_d_async()
    got = K.host_d(STEPS)
    torch.cuda.synchronize()
    runs, dref, ties = reference(confs, n)
    hist[(n > 1, got)] = hist.get((n > 1, got), 0) + 1
    if int(count) != runs or got != dref:
        if ties:
            tie_bad += 1
        else:
            bad.append({"trial": trial, "n": n, "runs": [int(count), runs], "d": [got, dref]})
print(json.dumps({"trials": 400, "tau": TAU, "cost": [A, D, R, AL], "mismatches": len(bad), "near_tie_mismatches": tie_bad,
                  "first": bad[:3], "d_hist_multi": {str(k[1]): v for k, v in sorted(hist.items()) if k[0]},
                  "d_hist_single": {str(k[1]): v for k, v in sorted(hist.items()) if not k[0]}}))
sys.exit(1 if bad else 0)
