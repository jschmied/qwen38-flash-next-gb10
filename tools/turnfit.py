# Per-turn TTFT over a cached prefix. (A) regression: 20k cached prefix + N fresh tokens, N in NS, 3 reps -> TTFT = a + N/b.
# (B) replay: one held-out trajectory, turn by turn (flattened transcript grows), max_tokens=1 -> TTFT vs new tokens per turn.
# (C) tokenize-only latency via /tokenize (chat template + tokenizer) for the largest prompt. Usage: turnfit.py <label> [profile_n]
import json, sys, time, urllib.request, random, glob, os, hashlib, statistics
URL="http://127.0.0.1:8092"; KEY=os.environ.get("OPENAI_API_KEY",""); H={"Content-Type":"application/json","Authorization":"Bearer "+KEY}
label=sys.argv[1]; PROF_N=int(sys.argv[2]) if len(sys.argv)>2 else 0
UNIT=("You are reviewing a large Python service. Here is the module under discussion. def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")
random.seed(7); WORDS="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda sigma tau upsilon omega parse route cache token block state layer kernel tensor".split()
def fresh(n_tokens):  # ~1 token per short word; distinct every call so it never hits the cache
    return " ".join(random.choice(WORDS)+str(random.randint(0,999)) for _ in range(max(1,int(n_tokens*0.55))))
def metrics():
    m={}
    for ln in urllib.request.urlopen(URL+"/metrics",timeout=30).read().decode().splitlines():
        if ln.startswith("#") or " " not in ln: continue
        k,v=ln.rsplit(" ",1); k=k.split("{")[0]
        try: m[k]=m.get(k,0.0)+float(v)
        except ValueError: pass
    return m.get("vllm:prefix_cache_hits_total",0.0), m.get("vllm:prefix_cache_queries_total",0.0)
def chat(content, max_tokens=1):
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":max_tokens,"messages":[{"role":"user","content":content}],"chat_template_kwargs":{"enable_thinking":False}}).encode()
    h0=metrics(); t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(urllib.request.Request(URL+"/v1/chat/completions",b,H),timeout=900).read()); el=time.perf_counter()-t0; h1=metrics()
    return el, d["usage"]["prompt_tokens"], h1[0]-h0[0], h1[1]-h0[1]
def tokenize(content):
    b=json.dumps({"model":"flashnext","messages":[{"role":"user","content":content}]}).encode()
    t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(urllib.request.Request(URL+"/tokenize",b,H),timeout=300).read()); return time.perf_counter()-t0, d.get("count")
def post(p): return urllib.request.urlopen(urllib.request.Request(URL+p,b"{}",H),timeout=600).status
# ---- (A) regression
base=UNIT*590  # ~20k tokens
el,pt,_,_=chat(base+"\nSummarise."); el2,_,h,q=chat(base+"\nSummarise."); print(f"  {label} A base: prompt_tokens={pt} cold={el:.2f}s second={el2:.2f}s hits/queries={h:.0f}/{q:.0f}", flush=True)
NS=[16,64,256,1024,4096]; rows=[]
for N in NS:
    ts=[]
    for r in range(3):
        el,pt,h,q=chat(base+"\n"+fresh(N)+"\nSummarise."); ts.append(el); rows.append((pt-20000 if pt>20000 else N, el))
    print(f"  {label} A N={N:5d}: prompt_tokens={pt} ttft median={statistics.median(ts):.3f}s all={' '.join(f'{x:.3f}' for x in ts)} hits/queries(last)={h:.0f}/{q:.0f}", flush=True)
# least squares on (new_tokens, ttft) using medians per N
import collections
byN=collections.defaultdict(list)
for n,e in rows: byN[n].append(e)
xs=[n for n in sorted(byN)]; ys=[statistics.median(byN[n]) for n in xs]
mx=sum(xs)/len(xs); my=sum(ys)/len(ys); b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); a=my-b*mx
print(f"  {label} A FIT: ttft = {a*1000:.0f} ms + new_tokens / {1/b:.0f} tok/s   (intercept = fixed per-request cost over a cached prefix)", flush=True)
# ---- (C) tokenize-only
tl,cnt=tokenize(base+"\n"+fresh(16)+"\nSummarise."); print(f"  {label} C tokenize(+template) of {cnt} tokens: {tl*1000:.0f} ms", flush=True)
tl,cnt=tokenize(fresh(16)); print(f"  {label} C tokenize(+template) of {cnt} tokens: {tl*1000:.0f} ms", flush=True)
# ---- (B) trajectory replay
fs=sorted(glob.glob('/opt/llm/swebench-runs/**/*.traj.json',recursive=True))
held=[f for f in fs if int(hashlib.md5((json.load(open(f)).get("instance_id","")).encode()).hexdigest(),16)%10==0]
d=json.load(open(held[0])); m=d["messages"]; sysmsg=next((x["content"] for x in m if x["role"]=="system"),"")[:1500]
lines=[]; prev_pt=0; turn=0
for i,x in enumerate(m[1:],1):
    r=x["role"]; c=x.get("content") or ""
    if r=="assistant":
        tc=x.get("tool_calls") or []; cmd=""
        for t in tc:
            try: cmd=json.loads(t["function"]["arguments"]).get("command","")
            except Exception: cmd=t["function"]["arguments"]
        lines.append(f"ASSISTANT:\n{(x.get('reasoning_content') or '')[:600]}\n```bash\n{cmd}\n```")
    elif r=="tool":
        lines.append(f"TOOL OUTPUT:\n{c[:3000]}"); turn+=1
        prompt=sysmsg+"\n\nTranscript so far:\n\n"+"\n\n".join(lines)+"\n\nContinue: think through the next step and give the next bash command."
        el,pt,h,q=chat(prompt); new=pt-prev_pt; prev_pt=pt
        print(f"  {label} B turn={turn:2d}: prompt_tokens={pt:6d} new={new:5d} ttft={el:.3f}s hits/queries={h:.0f}/{q:.0f}", flush=True)
        if turn>=24: break
    else: lines.append(f"USER:\n{c}")
# ---- profiled small turn
if PROF_N:
    chat(base+"\n"+fresh(PROF_N)+"\nSummarise.")
    print(f"  {label} P start_profile -> {post('/start_profile')}", flush=True)
    el,pt,h,q=chat(base+"\n"+fresh(PROF_N)+"\nSummarise."); print(f"  {label} P profiled turn N={PROF_N}: prompt_tokens={pt} ttft={el:.3f}s hits/queries={h:.0f}/{q:.0f}", flush=True)
    print(f"  {label} P stop_profile -> {post('/stop_profile')}", flush=True); time.sleep(15)
