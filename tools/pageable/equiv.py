# Generation equivalence (finding 226, s6). temp 0; hashes of the full completion text per request.
# A: 8 sequential prompts (c=1, MTP active). B: mixed batch, 2 decoding + 2 long prefills arriving mid-decode.
import json, sys, time, hashlib, threading, urllib.request
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
label = sys.argv[1]; corpus = open("/opt/llm/runners/corpus_frozen.txt").read()
def ask(text, max_tokens, out, key):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": text}],
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    d = json.loads(urllib.request.urlopen(r, timeout=900).read())
    c = d["choices"][0]["message"]["content"] or ""
    out[key] = (hashlib.sha256(c.encode()).hexdigest()[:12], d["usage"]["completion_tokens"])
res = {}
off = 1_000_000
for i in range(8):
    n = 6_000 + 4_000 * i
    ask(f"[eq-A-{i}]\n" + corpus[off:off + n] + "\nExplain what the code above does, step by step.", 200, res, f"A{i}")
    off += n
th = [threading.Thread(target=ask, args=(f"[eq-B-dec-{j}] Write a detailed essay about memory hierarchies in GPUs, part {j}.", 400, res, f"Bdec{j}")) for j in range(2)]
for t in th: t.start()
time.sleep(1.5)
th2 = [threading.Thread(target=ask, args=(f"[eq-B-pre-{j}]\n" + corpus[off + j * 70_000: off + (j + 1) * 70_000] + "\nSummarise the code above.", 64, res, f"Bpre{j}")) for j in range(2)]
for t in th2: t.start()
for t in th + th2: t.join()
print(json.dumps({"label": label, **{k: res[k] for k in sorted(res)}}))
