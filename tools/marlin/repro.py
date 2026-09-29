#!/usr/bin/env python3
"""Within-start reproducibility probe (2026-09-29, Marlin follow-up). Greedy 256 tokens, chat endpoint, thinking off.
Sequence: A, A, B, A with a unique `cache_salt` each (no prefix-cache hit: same tokens, cold path every time), then A
twice without a salt (the second one is a cache hit). Reports each request's hash and the first token-level divergence
from the first A. Prints ONE json object (armrun contract). argv: <tag>"""
import hashlib, json, sys, urllib.request, uuid
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"; tag = sys.argv[1]
A = ("Write a Python function that parses an ISO-8601 duration string such as P3DT4H12M into a number of seconds. "
     "Handle years and months as 365 and 30 days. Include a docstring and three doctest examples.")
B = "Explain in two short paragraphs why the sky looks blue at noon and red at sunset."


def chat(content, salt):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": 256, "messages": [{"role": "user", "content": content}],
         "chat_template_kwargs": {"enable_thinking": False}, "logprobs": True}
    if salt:
        b["cache_salt"] = salt
    r = urllib.request.Request(URL, json.dumps(b).encode(), {"Content-Type": "application/json",
                                                             "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
    toks = [t["token"] for t in d["choices"][0]["logprobs"]["content"]]
    return toks


seq = [("A", True), ("A", True), ("B", True), ("A", True), ("A", False), ("A", False)]
rows, ref = [], None
for name, salted in seq:
    toks = chat(A if name == "A" else B, uuid.uuid4().hex if salted else None)
    h = hashlib.sha256("\x00".join(toks).encode()).hexdigest()[:16]
    div = None
    if name == "A":
        if ref is None:
            ref = toks
        else:
            div = next((i for i, (x, y) in enumerate(zip(ref, toks)) if x != y), None)
    rows.append({"p": name, "salted": salted, "sha": h, "n": len(toks), "first_div_vs_A0": div})
a_hashes = [r["sha"] for r in rows if r["p"] == "A"]
print(json.dumps({"tag": tag, "A_classes": len(set(a_hashes)), "A0": a_hashes[0], "rows": rows}))
