# Send the agent loop's TURN-1 prompt (~7.5k tokens) N times, max_tokens=1, so the target's
# layer hashes over a LONG prefill can be compared across identical requests (cache off).
import json, sys, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY="sk-bench"
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220
label=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 3
msgs=[{"role":"user","content":SEED+"\nSummarise what handler() does in one sentence."}]
for i in range(n):
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":1,"logprobs":True,"top_logprobs":5,"messages":msgs,
                  "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    lp=d["choices"][0]["logprobs"]["content"][0]
    print(f"  {label} req{i+1}: prompt_tokens={d['usage']['prompt_tokens']} top={lp['token']!r} lp={lp['logprob']:.10g}", flush=True)
