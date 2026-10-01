#!/usr/bin/env python3
"""T12 gain bound: a concurrent round's forward (and its first MTP step) eager vs one CUDA graph replay, same inputs.
Wraps the names multi.py calls (compute, mtp_compute, MultiDecoder.round). Every EVERY-th round with >= 2 streams:
capture the round's forward as it stands, then time eager and replay calls in alternating blocks. Repeated calls rewrite
the same KV rows and advance DeltaNet states, so tokens after the first measured round are not meaningful; timing is
value-independent. Round walls are taken from unmeasured rounds only.
argv: <tensorfold src> <model dir> <streams> <tokens> [every]"""
import json, statistics, sys, threading, time
from pathlib import Path

src, model, streams, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
every = int(sys.argv[5]) if len(sys.argv) > 5 else 15
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda import multi                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

BLOCKS, CALLS = 3, 5
POOL = torch.cuda.graph_pool_handle()
LOG = {"walls": [], "fwd": [], "mtp": [], "mtp_steps": []}
state = {"round": 0, "measure": False, "mtp_done": False, "steps": 0}
_compute, _mtp_compute, _round = multi.compute, multi.mtp_compute, multi.MultiDecoder.round


def bound(fn, kind, rows, nseg):
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=POOL, capture_error_mode="thread_local"):
        fn()
    torch.cuda.synchronize()
    eager, replay = [], []
    for _ in range(BLOCKS):
        for arm, call in (("e", fn), ("r", g.replay)):
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(CALLS):
                call()
            torch.cuda.synchronize()
            (eager if arm == "e" else replay).append((time.perf_counter() - t) / CALLS * 1e3)
    del g
    LOG[kind].append({"round": state["round"], "streams": nseg, "rows": rows,
                      "eager_ms": [round(x, 3) for x in eager], "replay_ms": [round(x, 3) for x in replay]})


def compute(w, segs, b, **kw):
    out = _compute(w, segs, b, **kw)
    if state["measure"]:
        bound(lambda: _compute(w, segs, b, **kw), "fwd", segs[-1][2], len(segs))
    return out


def mtp_compute(w, segs, b, **kw):
    out = _mtp_compute(w, segs, b, **kw)
    state["steps"] += 1
    if state["measure"] and not state["mtp_done"]:
        state["mtp_done"] = True
        res = out.clone()
        bound(lambda: _mtp_compute(w, segs, b, **kw), "mtp", segs[-1][2], len(segs))
        out.copy_(res)                          # the drafts this round picks from stay the eager ones
    return out


def round_(self, told=None):
    live = sum(1 for s in self.streams.values() if not s.done and not s.waiting)
    state["round"] += 1
    state["measure"] = live >= streams and state["round"] % every == 0
    state["mtp_done"] = False
    state["steps"] = 0
    t = time.perf_counter()
    try:
        return _round(self, told)
    finally:
        if not state["measure"] and live >= streams:
            torch.cuda.synchronize()
            LOG["walls"].append((time.perf_counter() - t) * 1e3)
            LOG["mtp_steps"].append(state["steps"])
        state["measure"] = False


multi.compute, multi.mtp_compute, multi.MultiDecoder.round = compute, mtp_compute, round_
tok = AutoTokenizer.from_pretrained(model)
TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "Explain in about 400 words why lighthouses were automated in the twentieth century.",
         "Write a Python class for a thread-safe bounded priority queue with blocking get and put, plus tests.",
         "Schreibe eine ausführliche Erklärung auf Deutsch, wie eine Wärmepumpe im Winter ein Haus heizt."]
prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False), add_special_tokens=False) for t in TASKS[:streams]]
eng = FlashNextEngine(model, max_len=16384, streams=streams)
for k in LOG:
    LOG[k].clear()
state["round"] = 0
errors = []


def one(i):
    try:
        eng.generate(prompts[i], n, None, lambda t: None, stop_eos=False)
    except BaseException as exc:                # noqa: BLE001
        errors.append(repr(exc))


th = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
[t.start() for t in th]
[t.join() for t in th]
if errors:
    raise SystemExit(f"errors: {errors}")


def summary(kind):
    rows = LOG[kind]
    if not rows:
        return None
    e = [statistics.median(r["eager_ms"]) for r in rows]
    r = [statistics.median(r["replay_ms"]) for r in rows]
    sign = all(max(x["replay_ms"]) < min(x["eager_ms"]) for x in rows)
    return {"n": len(rows), "eager_med": round(statistics.median(e), 3), "replay_med": round(statistics.median(r), 3),
            "saved_med": round(statistics.median([a - b for a, b in zip(e, r)]), 3), "every_block_faster": sign}


walls = LOG["walls"]
print(json.dumps({"streams": streams, "rounds": state["round"], "wall_med_ms": round(statistics.median(walls), 3),
                  "wall_n": len(walls), "mtp_steps_mean": round(statistics.mean(LOG["mtp_steps"]), 2), "fwd": summary("fwd"), "mtp": summary("mtp"), "detail": LOG["fwd"] + LOG["mtp"]}))
