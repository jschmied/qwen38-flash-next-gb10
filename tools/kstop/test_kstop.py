#!/usr/bin/env python3
"""FNKSTOP unit test on the GPU (no server): fn_kstop.update() inside a CUDA graph captured in an IF node on
fn_kstop.pred(), as the patch captures the drafter's decode-step graphs. Checks, against a Python reference, over many
random confidence sequences: (1) the body runs exactly while any real request is active, (2) d per request and d_max,
(3) graph-padding rows (index >= nreq) never keep the chain alive or raise d_max, (4) d_max reaches the host.
Prints ONE json; exit 1 on any mismatch."""
import json, os, random, sys
os.environ["FN_KSTOP"] = "1"
os.environ.setdefault("FN_KSTOP_TAU", "0.75")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import fn_kstop as K

dev = "cuda"
MAXR, V, STEPS, TAU = 8, 64, 7, K.TAU
K.set_nreq(1, MAXR, dev)
pred = K.pred(MAXR, dev)
step = torch.zeros((), dtype=torch.int64, device=dev)
logits = torch.zeros(MAXR, V, device=dev)
count = torch.zeros((), dtype=torch.int32, device=dev)


def logits_for(confs):
    """Rows whose softmax top-1 probability equals conf (one peaked token, the rest flat)."""
    out = torch.full((len(confs), V), 0.0)
    for i, c in enumerate(confs):
        c = min(max(c, 1.0 / V + 1e-4), 0.999)
        # p_top = e^a / (e^a + V - 1)  ->  a = ln(c (V-1) / (1 - c))
        import math
        out[i, 0] = math.log(c * (V - 1) / (1 - c))
    return out


def body():
    count.add_(1)
    K.update(logits, step)


# warmup outside capture, then capture inside the IF node (as cudagraph_utils does with the patch)
body()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
torch.cuda.synchronize()
with torch.cuda.graph(g, stream=s):
    g.begin_capture_to_if_node(pred)
    body()
    g.end_capture_to_conditional_node()
torch.cuda.synchronize()

rng = random.Random(0)
bad = []
for trial in range(300):
    nreq = rng.randint(1, MAXR - 1)
    def draw():  # keep clear of tau so float rounding in the peaked-logit construction cannot flip a comparison
        c = rng.random()
        return c if abs(c - TAU) > 0.01 else draw()
    confs = [[draw() for _ in range(MAXR)] for _ in range(STEPS)]
    K.set_nreq(nreq, MAXR, dev)
    count.zero_()
    # step 0 = the prefill draft (eager, always runs)
    step.fill_(0); logits.copy_(logits_for(confs[0]).to(dev)); K.update(logits, step)
    for j in range(1, STEPS):
        step.fill_(j); logits.copy_(logits_for(confs[j]).to(dev))
        g.replay()
    K.copy_d_async()
    dmax_host = K.host_d(STEPS)
    torch.cuda.synchronize()
    # reference
    d_ref, act_ref, runs_ref = [1] * nreq, [True] * nreq, 0
    for j in range(1, STEPS):
        if not any(act_ref):
            break
        runs_ref += 1
        for r in range(nreq):
            if act_ref[r]:
                if confs[j][r] >= TAU:
                    d_ref[r] = j + 1
                else:
                    act_ref[r] = False
    got_d = K._S["d"][:nreq].tolist()
    if int(count) != runs_ref or got_d != d_ref or dmax_host != max(d_ref):
        bad.append({"trial": trial, "nreq": nreq, "runs": [int(count), runs_ref], "d": [got_d, d_ref],
                    "dmax": [dmax_host, max(d_ref)]})
print(json.dumps({"trials": 300, "tau": TAU, "mismatches": len(bad), "first": bad[:3],
                  "hist": dict(sorted(K._S["hist"].items()))}))
sys.exit(1 if bad else 0)
