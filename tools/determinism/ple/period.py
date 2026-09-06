# PERIOD: N identical sequential requests; per-request signature from probe positions + sampled token. Identical
# signatures are grouped into classes; the class sequence exposes slot/placement cycling (e.g. A B C A B C = period 3).
import json, sys, urllib.request, hashlib
CH="http://127.0.0.1:8092/v1/chat/completions"; H={"Content-Type":"application/json","Authorization":"Bearer "+__import__("os").environ["OPENAI_API_KEY"]}
label=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 16
P=json.load(open("/opt/llm/runners/posdiv_prompts.json")); prompt=[p for p in P if p["target"]==1500][0]["prompt"]
def post(b): return json.loads(urllib.request.urlopen(urllib.request.Request(CH,json.dumps(b).encode(),H),timeout=900).read())
PROBES=[1,2,10,700,1459]
def top(pos):
    it=sorted(pos.items(), key=lambda kv: kv[1].get("rank",1e9)); return tuple((k, round(v["logprob"],3)) for k,v in it[:2])
sigs=[]; full=[]
for r in range(N):
    d=post({"model":"flashnext","temperature":0,"max_tokens":1,"messages":[{"role":"user","content":prompt}],"prompt_logprobs":5,"logprobs":True,"top_logprobs":3,"chat_template_kwargs":{"enable_thinking":False}})
    plp=d["prompt_logprobs"]; s=d["choices"][0]["logprobs"]["content"][0]
    sig=tuple(top(plp[p]) for p in PROBES)+((s["token"], round(s["logprob"],3)),)
    fh=hashlib.sha1(json.dumps([[ (k,round(v["logprob"],3)) for k,v in sorted(pos.items())] if pos else None for pos in plp]).encode()).hexdigest()[:8]
    sigs.append(sig); full.append(fh)
    print(f"  {label} r{r:02d}: full={fh} p1={sig[0][0]} p700={sig[3][0]} sampled={sig[-1]}", flush=True)
classes={}; seq=[]
for fh in full: seq.append(classes.setdefault(fh, chr(65+len(classes))))
print(f"  {label} FULL-VECTOR classes: {len(classes)} distinct / {N};  sequence: {' '.join(seq)}", flush=True)
classes={}; seq=[]
for s in sigs: seq.append(classes.setdefault(s, chr(65+len(classes))))
print(f"  {label} PROBE-SIG classes: {len(classes)} distinct / {N};  sequence: {' '.join(seq)}", flush=True)
hs=[]
for r in range(N):
    d=post({"model":"flashnext","temperature":0,"max_tokens":32,"messages":[{"role":"user","content":prompt}],"chat_template_kwargs":{"enable_thinking":False}})
    hs.append(hashlib.sha1((d["choices"][0]["message"].get("content") or "").encode()).hexdigest()[:6])
classes={}; seq=[]
for h in hs: seq.append(classes.setdefault(h, chr(65+len(classes))))
print(f"  {label} E2E32 classes: {len(classes)} distinct / {N};  sequence: {' '.join(seq)}", flush=True)
