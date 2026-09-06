# Draft-vocab cell: real held-out agent prompts, MTP-3 at concurrency c; acceptance from /metrics deltas, tok/s from completion tokens / wall.
# Usage: dvcell.py <label> <c> <reps> <prompts.json>
import json, os, sys, time, threading, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY=os.environ.get("OPENAI_API_KEY",""); MET="http://127.0.0.1:8092/metrics"
label, c, reps, pf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; NEW=200
P=json.load(open(pf)); DUMP=open(f"/opt/llm/runners/results/dv-{label}.jsonl","a")
def metrics():
    m={}
    for ln in urllib.request.urlopen(MET,timeout=30).read().decode().splitlines():
        if ln.startswith("#") or " " not in ln: continue
        k,v=ln.rsplit(" ",1); k=k.split("{")[0]
        try: m[k]=m.get(k,0.0)+float(v)
        except ValueError: pass
    g=lambda *ks: next((m[k] for k in ks if k in m),0.0)
    return (g("vllm:spec_decode_num_draft_tokens_total","vllm:spec_decode_num_draft_tokens"), g("vllm:spec_decode_num_accepted_tokens_total","vllm:spec_decode_num_accepted_tokens"), g("vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_drafts"))
def garbage(t):
    w=t.split(); na=sum(1 for ch in t if ord(ch)>127)/max(len(t),1)
    return na>0.05 or (len(w)>20 and len(set(w))<0.4*len(w))
def one(out, pi, bar):
    p=P[pi % len(P)]
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":NEW,"ignore_eos":True,"messages":[{"role":"user","content":p["prompt"]}]}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    bar.wait(); t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=900).read()); el=time.perf_counter()-t0
    ch=d["choices"][0]["message"]; txt=(ch.get("reasoning_content") or ch.get("reasoning") or "")+(ch.get("content") or "")
    out.append({"prompt":pi,"elapsed":el,"completion_tokens":d["usage"]["completion_tokens"],"prompt_tokens":d["usage"]["prompt_tokens"],"text":txt})
k=0
for i in range(reps):
    m0=metrics(); outs=[[] for _ in range(c)]; bar=threading.Barrier(c)
    th=[threading.Thread(target=one,args=(outs[j],k+j,bar)) for j in range(c)]; k+=c
    t0=time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall=time.perf_counter()-t0; m1=metrics()
    dDT,dA,dN=(m1[x]-m0[x] for x in range(3)); ct=sum(o[0]["completion_tokens"] for o in outs)
    acc=f"accept={100*dA/dDT:.1f}% AL={1+dA/dN:.2f}" if dDT>0 and dN>0 else ("no-spec" if dDT==0 else f"accept={100*dA/dDT:.1f}%")
    g=sum(1 for o in outs if garbage(o[0]["text"]))
    print(f"  {label} DV c={c} rep={i}: {acc} | {ct} tok in {wall:.1f}s = {ct/wall:.1f} tok/s agg, {ct/c/wall:.1f} per stream | garbage {g}/{c}", flush=True)
    DUMP.write(json.dumps({"label":label,"c":c,"rep":i,"draft":dDT,"accepted":dA,"drafts":dN,"wall":wall,"reqs":[o[0] for o in outs]})+"\n"); DUMP.flush()
