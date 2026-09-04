#!/usr/bin/env python3
"""Warm-turn decomposition on a prefix-cache HIT: TTFT (streamed first token) as a function of the number of NEW
tokens appended to a cached ~7.5k-token prefix (0 = identical request, 1, 130, 1000), each 3x interleaved, plus one
128-token decode per size to get ms/token. Intercept = fixed hit cost, slope = prefill, decode = decode.
Usage: hitprobe.py <label>   (server on :8092, prefix caching on)."""
import json, sys, time, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY="sk-bench"; label=sys.argv[1]
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220  # ~7.5k tok
FILL="Additional context line about the dispatch path and its error handling. "  # ~12 tok
def req(new_tokens, max_tokens, nonce):
    extra = (FILL * max(1, new_tokens // 12))[: new_tokens * 5] if new_tokens else ""
    msgs=[{"role":"user","content":SEED+"\nSummarise what handler() does in one sentence."},
          {"role":"assistant","content":"handler builds a context from the request and dispatches it."},
          {"role":"user","content":(f"[{nonce}] " if new_tokens else "")+extra+" Name one more risk, briefly."}]
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":max_tokens,"stream":True,"messages":msgs,
                  "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t0=time.perf_counter(); ttft=None; n=0
    with urllib.request.urlopen(r,timeout=900) as resp:
        for line in resp:
            if not line.startswith(b"data:") or line.strip()==b"data: [DONE]": continue
            d=json.loads(line[5:]); ch=d["choices"][0] if d.get("choices") else None
            if ch and ch.get("delta",{}).get("content"):
                if ttft is None: ttft=time.perf_counter()-t0
                n+=1
            if d.get("usage"): pt=d["usage"].get("prompt_tokens")
    return ttft or (time.perf_counter()-t0), time.perf_counter()-t0, n
# cold prefix
t,tot,_=req(0,1,0); print(f"  {label} cold seed: ttft {t:6.2f} s", flush=True)
t,tot,_=req(0,1,0); print(f"  {label} 2nd identical (align-mode first repetition): ttft {t:6.2f} s", flush=True)
res={}
for rep in range(3):
    for new in (0,1,130,1000):
        t,tot,_=req(new,1,1000*rep+new); res.setdefault(new,[]).append(t)
        print(f"  {label} hit +{new:4d} new tokens rep {rep}: ttft {t:6.3f} s", flush=True)
for new in (0,1,130,1000):
    v=sorted(res[new]); print(f"  {label} HIT +{new:4d}: median ttft {v[1]:6.3f} s  (all {' '.join(f'{x:.3f}' for x in res[new])})", flush=True)
for new in (130,):
    for rep in range(2):
        t,tot,n=req(new,128,7000+rep); dec=(tot-t)/max(1,n-1)*1000
        print(f"  {label} DECODE after +{new} hit rep {rep}: ttft {t:6.3f} s  total {tot:6.2f} s  {n} tok  {dec:6.1f} ms/tok", flush=True)
