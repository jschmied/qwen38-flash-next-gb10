# Decode-only tok/s via streaming: time-to-first-token subtracted. Real held-out agent prompts, MTP-3. Usage: dvrate.py <label> <c> <reps> <prompts.json> [max_tokens]
import json, os, sys, time, threading, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; KEY=os.environ.get("OPENAI_API_KEY",""); MET="http://127.0.0.1:8092/metrics"
label, c, reps, pf = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; NEW=int(sys.argv[5]) if len(sys.argv)>5 else 400
P=json.load(open(pf)); DUMP=open(f"/opt/llm/runners/results/dvrate-{label}.jsonl","a")
def metrics():
    m={}
    for ln in urllib.request.urlopen(MET,timeout=30).read().decode().splitlines():
        if ln.startswith("#") or " " not in ln: continue
        k,v=ln.rsplit(" ",1); k=k.split("{")[0]
        try: m[k]=m.get(k,0.0)+float(v)
        except ValueError: pass
    g=lambda *ks: next((m[k] for k in ks if k in m),0.0)
    return (g("vllm:spec_decode_num_draft_tokens_total"), g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total"))
def one(out, pi, bar):
    p=P[pi % len(P)]
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":NEW,"ignore_eos":True,"stream":True,"stream_options":{"include_usage":True},"messages":[{"role":"user","content":p["prompt"]}]}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    bar.wait(); t0=time.perf_counter(); first=None; n=0; usage=None
    with urllib.request.urlopen(r,timeout=900) as resp:
        for raw in resp:
            line=raw.decode().strip()
            if not line.startswith("data:"): continue
            d=line[5:].strip()
            if d=="[DONE]": break
            j=json.loads(d)
            if j.get("usage"): usage=j["usage"]
            ch=j.get("choices") or []
            if ch and (ch[0].get("delta") or {}):
                delta=ch[0]["delta"]
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if first is None: first=time.perf_counter()
    end=time.perf_counter(); ct=(usage or {}).get("completion_tokens",0)
    out.append({"prompt":pi,"ttft":(first or end)-t0,"total":end-t0,"completion_tokens":ct,"decode_s":end-(first or end)})
k=0
for i in range(reps):
    m0=metrics(); outs=[[] for _ in range(c)]; bar=threading.Barrier(c)
    th=[threading.Thread(target=one,args=(outs[j],k+j,bar)) for j in range(c)]; k+=c
    [t.start() for t in th]; [t.join() for t in th]; m1=metrics()
    dDT,dA,dN=(m1[x]-m0[x] for x in range(3)); ct=sum(o[0]["completion_tokens"] for o in outs); dec=max(o[0]["decode_s"] for o in outs)
    per=[o[0]["completion_tokens"]/o[0]["decode_s"] for o in outs if o[0]["decode_s"]>0]
    acc=f"accept={100*dA/dDT:.1f}% AL={1+dA/dN:.2f}" if dDT>0 and dN>0 else "no-spec"
    print(f"  {label} RATE c={c} rep={i}: {acc} | ttft med {sorted(o[0]['ttft'] for o in outs)[c//2]:.2f}s | decode-only {sum(per)/len(per):.1f} tok/s per stream (agg {ct/dec:.1f}) | {ct} tok", flush=True)
    DUMP.write(json.dumps({"label":label,"c":c,"rep":i,"draft":dDT,"accepted":dA,"drafts":dN,"reqs":[o[0] for o in outs]})+"\n"); DUMP.flush()
