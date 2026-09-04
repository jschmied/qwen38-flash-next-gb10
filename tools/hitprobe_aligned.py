#!/usr/bin/env python3
"""Companion to hitprobe.py: pads the cached prefix so it ends exactly on the 1,600-token align block boundary
(main build sets the attention/mamba block to 1,600 tokens), then measures hit TTFT for +1 / +130 new tokens.
If the warm-turn intercept is the recompute of the un-hit tail (prefix mod 1,600), it collapses here.
Usage: hitprobe_aligned.py <label> [block=1600]"""
import json, sys, time, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY="sk-bench"; label=sys.argv[1]; BLOCK=int(sys.argv[2]) if len(sys.argv)>2 else 1600
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220
FILL="Additional context line about the dispatch path and its error handling. "
def call(seed, tail, max_tokens, stream):
    msgs=[{"role":"user","content":seed+"\nSummarise what handler() does in one sentence."},
          {"role":"assistant","content":"handler builds a context from the request and dispatches it."},
          {"role":"user","content":tail}]
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":max_tokens,"stream":stream,"messages":msgs,
                  "chat_template_kwargs":{"enable_thinking":False}, **({"stream_options":{"include_usage":True}} if stream else {})}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
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
TAIL="Name one more risk, briefly."
# 1) measure prefix length with a 1-token tail request (prompt = prefix + tail); tail tokens = const
_, pt = call(SEED, TAIL, 1, False); _, pt2 = call(SEED+" x", TAIL, 1, False); print(f"  {label} probe: prompt_tokens {pt} (+' x' -> {pt2})", flush=True)
# the cached prefix is everything before the last user message; we cannot see it directly, so align the TOTAL prompt of the
# +0 request to the boundary and then also report +1/+130 relative to that
words = 0; seed = SEED
while True:
    _, pt = call(seed, TAIL, 1, False)
    r = pt % BLOCK
    if r == 0: break
    add = BLOCK - r
    seed = seed + (" x" * add)   # " x" is one token
    words += add
    if words > 4*BLOCK: print("  could not align", flush=True); break
print(f"  {label} aligned prompt_tokens={pt} (added {words} filler tokens)", flush=True)
res={}
for rep in range(3):
    for new in (0,1,130):
        tail = TAIL if new==0 else f"[{rep}{new}] " + (FILL*(new//12))[:new*5] + " " + TAIL
        t, p = call(seed, tail, 1, True); res.setdefault(new,[]).append(t)
        print(f"  {label} aligned hit +{new:4d} rep {rep}: ttft {t:6.3f} s  prompt_tokens {p}", flush=True)
for new in (0,1,130):
    v=sorted(res[new]); print(f"  {label} ALIGNED HIT +{new:4d}: median ttft {v[1]:6.3f} s  (all {' '.join(f'{x:.3f}' for x in res[new])})", flush=True)
