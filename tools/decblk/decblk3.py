#!/usr/bin/env python3
"""Decode-written cache block probe v3: 380 numbers (~3.1k-token prompt) so the 520 sampled tokens cross the
3456-token scheduler block (hits come in 3456-token units on this model; v2's 1871-token follow-up could never hit).
v2 text follows. Decode-written cache block probe v2 (vllm#53912 design). v1 used /v1/completions with token-id prompts and got 0
prefix hits for the whole run, so v2 stays on the chat endpoint and first records whether each endpoint hits at all.
Per seed: a user message of random 7-digit numbers (~1.4k tokens) and a partial assistant message; a sampled
continuation (T=1.0: low draft acceptance) of 520 tokens crosses the 1728-token block. The follow-up re-sends the same
partial assistant message extended by the first 480 generated tokens' text (continue_final_message, so both requests
render through the same template path). B reads the cache and must hit past the first request's prompt (into
decode-written blocks); R skips it (prompt_logprobs=1). Greedy 24 tokens, compared. Prints ONE json. argv: <tag>"""
import json, random, sys, urllib.request

URL = "http://127.0.0.1:8092"
H = {"Content-Type": "application/json", "Authorization": "Bearer none"}


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), H)
    return json.loads(urllib.request.urlopen(req, timeout=900).read())


def hits():
    text = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    return sum(float(l.rsplit(" ", 1)[1]) for l in text.splitlines() if l.startswith("vllm:prefix_cache_hits_total"))


def chat(msgs, n, **kw):
    body = {"model": "flashnext", "messages": msgs, "max_tokens": n, "chat_template_kwargs": {"enable_thinking": False}}
    body.update(kw)
    h0 = hits()
    d = post("/v1/chat/completions", body)
    return d["choices"][0]["message"]["content"] or "", d["usage"]["prompt_tokens"], hits() - h0


diag = {}
for ep in ("chat", "completions"):
    txt = f"[diag {ep}] " + " ".join(str(i) for i in range(900))
    for rep in (0, 1):
        if ep == "chat":
            _, _, h = chat([{"role": "user", "content": txt}], 1, temperature=0)
        else:
            h0 = hits()
            post("/v1/completions", {"model": "flashnext", "prompt": txt, "max_tokens": 1})
            h = hits() - h0
        diag[f"{ep}_rep{rep}_hits"] = h

rows = []
head = "Continuing the list:\n"
for s in range(1, 17):
    rnd = random.Random(s)
    u = ("Numbers:\n" + " ".join(str(rnd.randint(1000000, 9999999)) for _ in range(380))
         + "\nContinue this list with 150 more numbers.")
    cont = dict(continue_final_message=True, add_generation_prompt=False)
    g, p0, _ = chat([{"role": "user", "content": u}, {"role": "assistant", "content": head}], 520,
                    temperature=1.0, seed=s, ignore_eos=True, **cont)
    tok = post("/tokenize", {"model": "flashnext", "prompt": g, "add_special_tokens": False})["tokens"]
    cut = post("/detokenize", {"model": "flashnext", "tokens": tok[:480]})["prompt"]
    msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": head + cut}]
    b, pb, hb = chat(msgs, 24, temperature=0, **cont)
    r, pr, hr = chat(msgs, 24, temperature=0, prompt_logprobs=1, **cont)
    rows.append({"seed": s, "prompt0": p0, "follow": pb, "hits_B": hb, "hits_R": hr, "equal": b == r,
                 "first_div": next((i for i, (x, y) in enumerate(zip(b, r)) if x != y), None)})

print(json.dumps({"diag": diag, "n": len(rows), "divergent": sum(not x["equal"] for x in rows),
                  "B_hit_past_prompt": all(x["hits_B"] > x["prompt0"] for x in rows),
                  "B_hit_min": min(x["hits_B"] for x in rows), "R_miss_all": all(x["hits_R"] == 0 for x in rows),
                  "prompt0": [rows[0]["prompt0"], rows[-1]["prompt0"]], "rows": rows}))
