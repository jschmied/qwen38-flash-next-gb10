#!/usr/bin/env python3
"""Does the PREFILL FORWARD PASS already diverge? ~1 s, no generation.

Sends one fixed prompt N times with max_tokens=1 and logprobs, and compares the top-k
logprob vector of the FIRST token. Nothing is generated, so sampling, speculative decoding
and feedback are all excluded by construction.

  logprobs identical  -> prefill is deterministic; the source is downstream (decode loop,
                         spec-decode rollback, sampling)
  logprobs differ     -> the source is in the prefill forward pass itself
"""
import json, sys, hashlib, urllib.request

URL = "http://127.0.0.1:8092/v1/chat/completions"
KEY = "sk-bench"
PROMPT = ("Write a detailed technical explanation of how a copy-on-write page table works in a "
          "modern operating system kernel, covering fork(), page faults, reference counting, and "
          "the interaction with the TLB. Be thorough and precise.")

label = sys.argv[1] if len(sys.argv) > 1 else "?"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
# 3rd arg: max_tokens. >1 makes the server run decode steps as well as the prefill, so the
# layer-hash hooks capture decode passes too. Needed to bisect the generation-side source,
# which survives --no-enable-prefix-caching and is therefore NOT the prefill defect.
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 1
sigs, tops = [], []
_all_persig = []
for i in range(n):
    body = json.dumps({
        "model": "flashnext", "temperature": 0, "max_tokens": mx, "logprobs": True,
        "top_logprobs": 20, "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(
        URL, body, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except Exception as e:
        print(f"  LOGIT {label} req{i+1}: FAILED: {e}", flush=True); continue
    ch = d["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    if not lp:
        print(f"  LOGIT {label} req{i+1}: no logprobs returned (server may not expose them)", flush=True); continue
    top = lp[0].get("top_logprobs") or []
    # per-token signatures over EVERY generated token, not just lp[0]. Until 2026-09-02 this
    # hashed only the first token, so with max_tokens>1 every decode step went unchecked and
    # "1 distinct of 3" meant only "prefill deterministic". GENBIS2's layer hashes showed decode
    # diverging from layer 1 while this probe reported identical output -- the probe was blind.
    def _sig(tk):
        return hashlib.sha256("|".join(f"{t.get('token')!r}:{t.get('logprob'):.12g}"
                                       for t in (tk.get("top_logprobs") or [])).encode()).hexdigest()[:12]
    persig = [_sig(tk) for tk in lp]
    sig = hashlib.sha256("|".join(persig).encode()).hexdigest()[:12]
    sigs.append(sig); tops.append(top)
    if len(persig) > 1:
        print(f"  LOGIT {label} req{i+1} per-token: " + " ".join(persig), flush=True)
    _all_persig.append(persig)
    best = top[0] if top else {}
    print(f"  LOGIT {label} req{i+1}: top={best.get('token')!r} lp={best.get('logprob'):.10g} sig={sig}", flush=True)

u = len(set(sigs))
if _all_persig and len({len(x) for x in _all_persig}) == 1 and len(_all_persig[0]) > 1:
    first = next((k for k in range(len(_all_persig[0])) if len({x[k] for x in _all_persig}) > 1), None)
    if first is None:
        print(f"  LOGIT {label} PER-TOKEN: all {len(_all_persig[0])} tokens identical across {len(_all_persig)} requests", flush=True)
    elif first == 0:
        print(f"  LOGIT {label} PER-TOKEN: token 1 (prefill) already differs", flush=True)
    else:
        print(f"  LOGIT {label} PER-TOKEN: prefill identical; first divergent token = {first+1} (decode step {first})", flush=True)
if u:
    print(f"  LOGIT {label} RESULT: {u} distinct of {len(sigs)}  -> " +
          ("PREFILL DIVERGES (source is upstream of the decode loop)" if u > 1
           else "prefill deterministic -> source is downstream"))
    if u > 1 and len(tops) >= 2:
        a, b = tops[0], tops[1]
        for k, (x, y) in enumerate(zip(a, b)):
            if x.get("token") != y.get("token") or abs(x.get("logprob", 0) - y.get("logprob", 0)) > 0:
                print(f"    first difference at rank {k}: {x.get('token')!r} {x.get('logprob'):.10g}"
                      f"  vs  {y.get('token')!r} {y.get('logprob'):.10g}")
                break
