#!/usr/bin/env python3
"""Companion to hitprobe.py (v2). Aligns the SHARED prefix (seed turn + assistant turn + the next user header) to
the 1,600-token align block, so the new tokens of the next turn are the only un-hit part. The shared length is
prompt_tokens - 1 (one-token tail) - S (chat-template suffix after the tail, unknown ~4..12), so we sweep S: for
each candidate pad the seed accordingly, serve tail "a" (cold), then tail "b" (hit) and take the fastest — that
pad is aligned. Then +1 / +130 new tokens x3 on the aligned seed. v1 aligned the TOTAL prompt, which is the worst
case (the last block is always recomputed). Usage: hitprobe_aligned.py <label> [block=1600]"""
import json, sys, time, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY="sk-bench"; label=sys.argv[1]; BLOCK=int(sys.argv[2]) if len(sys.argv)>2 else 1600
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220
FILL="Additional context line about the dispatch path and its error handling. "
def call(seed, tail, max_tokens=1, stream=True):
    msgs=[{"role":"user","content":seed+"\nSummarise what handler() does in one sentence."},
          {"role":"assistant","content":"handler builds a context from the request and dispatches it."},
          {"role":"user","content":tail}]
    body={"model":"flashnext","temperature":0,"max_tokens":max_tokens,"stream":stream,"messages":msgs,"chat_template_kwargs":{"enable_thinking":False}}
    if stream: body["stream_options"]={"include_usage":True}
    r=urllib.request.Request(URL,json.dumps(body).encode(),{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t0=time.perf_counter()
    if not stream:
        d=json.loads(urllib.request.urlopen(r,timeout=900).read()); return time.perf_counter()-t0, d["usage"]["prompt_tokens"]
    ttft=None; pt=None
    with urllib.request.urlopen(r,timeout=900) as resp:
        for line in resp:
            if not line.startswith(b"data:") or line.strip()==b"data: [DONE]": continue
            d=json.loads(line[5:]); ch=d["choices"][0] if d.get("choices") else None
            if ch and ch.get("delta",{}).get("content") and ttft is None: ttft=time.perf_counter()-t0
            if d.get("usage"): pt=d["usage"].get("prompt_tokens")
    return ttft or (time.perf_counter()-t0), pt
# filler unit: 1 token each?
_, p0 = call(SEED, "a", stream=False); _, p4 = call(SEED+" x"*4, "a", stream=False); unit_tok=(p4-p0)/4
print(f"  {label} prompt_tokens(tail 'a') = {p0}; ' x' = {unit_tok:.2f} tok/unit", flush=True)
assert unit_tok==1.0, "filler is not 1 token"
best=None
for S in range(2, 14):
    shared = p0 - 1 - S
    pad = (BLOCK - shared % BLOCK) % BLOCK
    seed = SEED + " x"*pad
    tc,_ = call(seed, "a"); th,pt = call(seed, "b")
    print(f"  {label} S={S:2d} pad={pad:4d} prompt_tokens={pt}: cold {tc:5.2f} s  hit(tail 'b') {th:6.3f} s", flush=True)
    if best is None or th < best[0]: best=(th,S,pad,seed)
th,S,pad,seed=best; print(f"  {label} BEST: S={S} pad={pad} -> hit with 1 new token {th:.3f} s (shared prefix on the {BLOCK} boundary)", flush=True)
res={}
for rep in range(3):
    for new in (1,130):
        tail = f"c{rep}" if new==1 else f"[{rep}] " + (FILL*(new//12))[:new*5] + " end"
        t,pt = call(seed, tail); res.setdefault(new,[]).append(t)
        print(f"  {label} aligned hit +{new:4d} rep {rep}: ttft {t:6.3f} s  prompt_tokens {pt}", flush=True)
for new in (1,130):
    v=sorted(res[new]); print(f"  {label} ALIGNED HIT +{new:4d}: median ttft {v[1]:6.3f} s  (all {' '.join(f'{x:.3f}' for x in res[new])})", flush=True)
