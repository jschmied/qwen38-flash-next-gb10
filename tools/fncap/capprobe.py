#!/usr/bin/env python3
"""FNCAP capture probe: real traffic (decprobe: c=1 essays, c=4, agent loop) while hooks capture, then trigger the
in-worker bench (BENCH file + one request), wait for BENCH_DONE, summarise. Prints ONE json.  argv: <tag>"""
import glob, json, os, subprocess, sys, time, urllib.request
D = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("CAPDIR", "/opt/llm/capture/fncap-0924")
tag = sys.argv[1]
t0 = time.time()
p = subprocess.run([sys.executable, "/opt/llm/runners/fncap/decprobe.py", tag], capture_output=True, text=True)
traffic = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1]) if "{" in p.stdout else {"err": p.stderr[-300:]}
open(os.path.join(D, "BENCH"), "w").write("1\n")
b = json.dumps({"model": "flashnext", "messages": [{"role": "user", "content": "Count from one to twenty."}],
                "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode()
r = urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", b,
                           {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"})
try:
    urllib.request.urlopen(r, timeout=1800).read()
except Exception as ex:
    print(f"bench request: {ex}", file=sys.stderr)
for _ in range(360):
    if os.path.exists(os.path.join(D, "BENCH_DONE")):
        break
    time.sleep(5)
bench = json.load(open(os.path.join(D, "bench.json"))) if os.path.exists(os.path.join(D, "bench.json")) else []
dirs = [d for d in glob.glob(os.path.join(D, "*")) if os.path.isdir(d)]
res = dict(capture_dirs=len(dirs), dec_files=len(glob.glob(os.path.join(D, "*", "dec*.pt"))),
           routing=os.path.exists(os.path.join(D, "routing.pt")), bench_entries=len(bench),
           bench_errors=sum(1 for x in bench if "error" in x), c1_ms_tok=traffic.get("c1_ms_tok"),
           secs=round(time.time() - t0))
print(json.dumps(res))
