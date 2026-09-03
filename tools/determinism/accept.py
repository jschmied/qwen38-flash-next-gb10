#!/usr/bin/env python3
"""Cumulative speculative-decode acceptance from /metrics (server lifetime). Usage: accept.py <label>."""
import sys, urllib.request
raw=urllib.request.urlopen("http://127.0.0.1:8092/metrics",timeout=30).read().decode(); m={}
for ln in raw.splitlines():
    if ln.startswith("#") or " " not in ln: continue
    k,v=ln.rsplit(" ",1); k=k.split("{")[0]
    try: m[k]=m.get(k,0.0)+float(v)
    except ValueError: pass
def g(*ks):
    for k in ks:
        if k in m: return m[k]
    return 0.0
D=g("vllm:spec_decode_num_drafts_total","vllm:spec_decode_num_drafts"); DT=g("vllm:spec_decode_num_draft_tokens_total","vllm:spec_decode_num_draft_tokens"); A=g("vllm:spec_decode_num_accepted_tokens_total","vllm:spec_decode_num_accepted_tokens")
label=sys.argv[1] if len(sys.argv)>1 else ""
if DT>0 and D>0: print(f"  ACCEPT {label}: drafts={D:.0f} draft_tok={DT:.0f} accepted={A:.0f} rate={100*A/DT:.1f}%  mean_accept_len={1+A/D:.2f}")
else: print(f"  ACCEPT {label}: no draft tokens recorded (spec off, or counters absent)")
