#!/usr/bin/env python3
"""EOS-correct agent-loop probe with per-turn speculative-decode acceptance (rewritten 2026-09-03
after the /tmp loss; successor of tools/agentloop.py).

Finding 59: `ignore_eos: true` on a chat model measures post-EOS filler (a repeat the drafter
predicts ~100 % or an `<|im_start|>` wall it predicts 0 %), not the model. Default here is to STOP
AT EOS (AGENT_EOS=1); set AGENT_EOS=0 to reproduce the old pinned-work loop. Reports tokens,
ms/tok over the real tokens, prefix-cache hits per turn, and per-turn acceptance from the
/metrics counter deltas (never from `usage`)."""
import json, os, sys, time, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; MET="http://127.0.0.1:8092/metrics"; KEY="sk-bench"
EOS = os.environ.get("AGENT_EOS", "1") == "1"
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220  # ~7.5k tok
def metrics():
    raw=urllib.request.urlopen(MET,timeout=30).read().decode(); m={}
    for ln in raw.splitlines():
        if ln.startswith("#") or " " not in ln: continue
        k,v=ln.rsplit(" ",1); k=k.split("{")[0]
        try: m[k]=m.get(k,0.0)+float(v)
        except ValueError: pass
    return m
def g(m,*keys):
    for k in keys:
        if k in m: return m[k]
    return 0.0
def spec(m):
    return (g(m,"vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_drafts"),
            g(m,"vllm:spec_decode_num_draft_tokens_total","vllm:spec_decode_num_draft_tokens"),
            g(m,"vllm:spec_decode_num_accepted_tokens_total","vllm:spec_decode_num_accepted_tokens"))
def turn(msgs):
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":130,"ignore_eos":(not EOS),"messages":msgs,
                  "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    return (time.perf_counter()-t0, d["choices"][0]["message"].get("content") or "",
            d.get("usage",{}).get("completion_tokens",0))
label=sys.argv[1]
msgs=[{"role":"user","content":SEED+"\nSummarise what handler() does in one sentence."}]
m0=metrics(); tot=0.0; TOKS=0; D=DT=A=0.0
for i in range(8):
    dt,out,tk=turn(msgs); tot+=dt; TOKS+=tk
    msgs.append({"role":"assistant","content":out})
    msgs.append({"role":"user","content":f"Turn {i+2}: name one more risk in this code, briefly."})
    m1=metrics(); d0,t0_,a0=spec(m0); d1,t1,a1=spec(m1); dd,ddt,da=d1-d0,t1-t0_,a1-a0; D+=dd; DT+=ddt; A+=da
    acc=f"acc {100*da/ddt:5.1f}% len {1+da/dd:4.2f}" if ddt>0 and dd>0 else "acc  n/a"
    print(f"  {label} turn {i+1}: {dt:6.2f} s  {tk:4d} tok  hits+{g(m1,'vllm:prefix_cache_hits_total')-g(m0,'vllm:prefix_cache_hits_total'):7.0f}  {acc}", flush=True)
    m0=m1
per_tok = tot/TOKS*1000 if TOKS else float("nan")
flag = "" if EOS or TOKS == 8*130 else f"  !! UNEQUAL WORK (expected {8*130})"
print(f"  {label} TOTAL {tot:.1f} s over 8 turns  ({tot/8:.2f} s/turn)  {TOKS} tok  {per_tok:.2f} ms/tok{flag}")
if DT>0 and D>0: print(f"  ACCEPT {label}-post: drafts={D:.0f} draft_tok={DT:.0f} accepted={A:.0f} rate={100*A/DT:.1f}%  mean_accept_len={1+A/D:.2f}")
else: print(f"  ACCEPT {label}-post: no draft tokens recorded (spec off, or counters absent)")
