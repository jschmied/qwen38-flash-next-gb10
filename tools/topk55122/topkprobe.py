#!/usr/bin/env python3
"""#55122 server probe: cold TTFT at ~8k and ~30k (prefix cache defeated by a unique salt at the front of every
prompt), then the 8-turn agent loop (agentloop2.py). Also records which top-k path the worker logged and which
_C_det library each process has mapped. Prints ONE json object (armrun contract).  argv: <arm tag>"""
import json, os, re, statistics, subprocess, sys, time, urllib.request, uuid
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
UNIT = ("You are reviewing a large Python service. Here is the module under discussion. "
        "def handler(req):\n    ctx = build_context(req)\n    return dispatch(ctx)\n")
tag = sys.argv[1]
res = {}
for reps, name in ((220, "8k"), (860, "30k")):
    ts = []; ptok = 0
    for i in range(3):
        salt = f"[session {uuid.uuid4().hex}]\n"
        msgs = [{"role": "user", "content": salt + UNIT * reps + "\nSummarise what handler() does in one sentence."}]
        b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 1, "messages": msgs,
                        "chat_template_kwargs": {"enable_thinking": False}}).encode()
        r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
        t0 = time.perf_counter(); d = json.loads(urllib.request.urlopen(r, timeout=900).read())
        ts.append(round(time.perf_counter() - t0, 3)); ptok = d["usage"]["prompt_tokens"]
    res[f"ttft{name}_med"] = statistics.median(ts); res[f"ttft{name}_all"] = ts; res[f"ptok{name}"] = ptok
p = subprocess.run([sys.executable, "/opt/llm/runners/agentloop2.py", tag], capture_output=True, text=True)
m = re.search(r"TOTAL ([\d.]+) s over 8 turns\s+\(([\d.]+) s/turn\)\s+(\d+) tok\s+([\d.]+) ms/tok", p.stdout)
if m:
    res.update(total_s=float(m[1]), s_per_turn=float(m[2]), tokens=int(m[3]), ms_per_tok=float(m[4]))
a = re.search(r"rate=([\d.]+)%\s+mean_accept_len=([\d.]+)", p.stdout)
if a:
    res.update(accept_rate=float(a[1]), accept_len=float(a[2]))
res["agentloop_raw"] = [l for l in p.stdout.splitlines() if "turn" in l or "TOTAL" in l or "ACCEPT" in l]
try:
    lg = open(f"/opt/llm/armrun-{tag}.log", errors="replace").read()
    res["env_path"] = sorted(set(re.findall(r"QSATOPK env path=(\S+)", lg)))
    res["call_path"] = sorted(set(re.findall(r"QSATOPK call path=(\S+)", lg)))
except FileNotFoundError:
    res["env_path"] = res["call_path"] = ["<no log>"]
mapped = set()
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        for ln in open(f"/proc/{pid}/maps"):
            if "_C_det" in ln:
                mapped.add(ln.split()[-1])
    except OSError:
        pass
res["det_libs_mapped"] = sorted(mapped)
print(json.dumps(res))
