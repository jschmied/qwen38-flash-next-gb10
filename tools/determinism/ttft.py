# TTFT (max_tokens=1 wall time) for prompts of ~8k and ~30k tokens, 3 requests each, cache off.
import json, sys, time, urllib.request, statistics
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY="sk-bench"
UNIT=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")   # ~34 tokens
label=sys.argv[1]
for reps,name in ((220,"8k"),(860,"30k")):
    msgs=[{"role":"user","content":UNIT*reps+"\nSummarise what handler() does in one sentence."}]
    ts=[]; ptok=0
    for i in range(3):
        b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":1,"messages":msgs,"chat_template_kwargs":{"enable_thinking":False}}).encode()
        r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
        t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=900).read()); ts.append(time.perf_counter()-t0)
        ptok=d["usage"]["prompt_tokens"]
    print(f"  {label} TTFT {name}: prompt_tokens={ptok} median={statistics.median(ts):.2f}s  all={' '.join(f'{x:.2f}' for x in ts)}", flush=True)
