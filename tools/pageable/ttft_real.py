# TTFT (max_tokens=1) on DIVERSE text: distinct n-grams per request, distinct slice per request, so neither the
# prefix cache nor a repeated unit hides PLE row reads. Corpus = the venv's vllm sources, sorted, deterministic.
import json, sys, time, urllib.request, glob, os
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
label = sys.argv[1]
root = "/opt/llm/runtime/vllm-venv-fnmain3/lib/python3.12/site-packages/vllm"
files = sorted(glob.glob(root + "/**/*.py", recursive=True))
corpus = "".join(open(f, errors="ignore").read() for f in files)
off = 0
res = {"label": label}
for name, chars in (("8k", 26_000), ("30k", 98_000)):
    rows = []
    for i in range(3):
        text = corpus[off:off + chars]; off += chars
        msgs = [{"role": "user", "content": f"[{label}-{name}-{i}]\n" + text + "\nOne word."}]
        b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 1, "messages": msgs,
                        "chat_template_kwargs": {"enable_thinking": False}}).encode()
        r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        t0 = time.perf_counter(); d = json.loads(urllib.request.urlopen(r, timeout=900).read())
        rows.append((round(time.perf_counter() - t0, 2), d["usage"]["prompt_tokens"]))
    res[name] = rows
    print(f"  {label} TTFT {name}: " + " ".join(f"{t:.2f}s/{p}tok" for t, p in rows), flush=True)
print(json.dumps(res))
