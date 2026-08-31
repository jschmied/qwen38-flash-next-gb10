#!/usr/bin/env python3
"""Agent-shaped latency probe: 8 dependent turns over a growing conversation.

WHY ignore_eos IS NOT OPTIONAL. max_tokens is a CEILING, not a target. Without
ignore_eos this script times whatever the model happened to emit, so an arm whose
turns stop at 40 tokens beats one whose turns run to 130 with no difference in
speed. That defect produced a full day of invalid agent-loop comparisons on
2026-08-31 -- including a "37% monotonic trend" that was withdrawn -- and it was
caught only because a replication of the SAME config gave 1.94 and then 1.43
s/turn.

So: the work per arm is pinned at 8 x 130 tokens, every turn's completion_tokens
is recorded, and the summary refuses to look clean if the total is not exactly
1040. Compare ms/tok, never s/turn across arms of unequal length.

Sanity check that this is measuring what you think: ms/tok should land within a
few percent of 1000/decode_tps from an independent benchmark. If it does not, the
harness and the engine disagree and one of them is wrong.
"""
# Does MTP cost more in prefix-cache hits than it gains in decode, on an AGENT-SHAPED
# workload? Every benchmark we have used UNIQUE prompts, so prefix reuse -- the thing
# an agent loop is made of -- has never been in any number we published.
# Simulates a conversation: a long shared prefix that grows by one short turn each step.
import json, sys, time, urllib.request
URL="http://127.0.0.1:8092/v1/chat/completions"; MET="http://127.0.0.1:8092/metrics"; KEY="sk-bench"
SEED=("You are reviewing a large Python service. Here is the module under discussion. "
      "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n") * 220  # ~8k tok
def hits():
    raw=urllib.request.urlopen(MET,timeout=30).read().decode()
    for ln in raw.splitlines():
        if ln.startswith("vllm:prefix_cache_hits_total"): return float(ln.rsplit(" ",1)[1])
    return 0.0
def turn(msgs):
    b=json.dumps({"model":"flashnext","temperature":0,"max_tokens":130,"ignore_eos":True,"messages":msgs,
                  "chat_template_kwargs":{"enable_thinking":False}}).encode()
    r=urllib.request.Request(URL,b,{"Content-Type":"application/json","Authorization":"Bearer "+KEY})
    t0=time.perf_counter(); d=json.loads(urllib.request.urlopen(r,timeout=900).read())
    return (time.perf_counter()-t0, d["choices"][0]["message"].get("content") or "",
            d.get("usage",{}).get("completion_tokens",0))
label=sys.argv[1]
msgs=[{"role":"user","content":SEED+"\nSummarise what handler() does in one sentence."}]
h0=hits(); tot=0.0
TOKS=0
for i in range(8):
    dt,out,tk=turn(msgs); tot+=dt; TOKS+=tk
    msgs.append({"role":"assistant","content":out})
    msgs.append({"role":"user","content":f"Turn {i+2}: name one more risk in this code, briefly."})
    print(f"  {label} turn {i+1}: {dt:6.2f} s  {tk:4d} tok  hits+{hits()-h0:8.0f}", flush=True); h0=hits()
# s/turn alone is meaningless if turns differ in length -- max_tokens is a ceiling.
# ignore_eos pins it, and the token count proves the arms did equal work.
per_tok = tot/TOKS*1000 if TOKS else float("nan")
flag = "" if TOKS == 8*130 else f"  !! UNEQUAL WORK (expected {8*130})"
print(f"  {label} TOTAL {tot:.1f} s over 8 turns  ({tot/8:.2f} s/turn)  "
      f"{TOKS} tok  {per_tok:.2f} ms/tok{flag}")
