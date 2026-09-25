import os, json, time, urllib.request, subprocess, collections
def snap():
    d = {}
    for p in os.listdir("/proc"):
        if not p.isdigit(): continue
        try:
            for t in os.listdir(f"/proc/{p}/task"):
                s = open(f"/proc/{p}/task/{t}/stat").read(); r = s.rsplit(")", 1)[1].split()
                name = s[s.find("(") + 1:s.rfind(")")]
                d[(int(p), int(t))] = (name, int(r[9]))           # majflt = field 12 -> index 9 after comm
        except Exception: pass
    return d
def comm(p):
    try: return open(f"/proc/{p}/comm").read().strip()
    except Exception: return "?"
a = snap(); t0 = time.time()
b = {"model": "flashnext", "temperature": 0, "max_tokens": 400, "messages": [{"role": "user", "content": "Write about 400 words on the history of the printing press."}], "chat_template_kwargs": {"enable_thinking": False}}
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", json.dumps(b).encode(), {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}), timeout=900).read()
dt = time.time() - t0; z = snap()
byproc = collections.Counter(); bythr = collections.Counter()
for k, (n, m) in z.items():
    d = m - a.get(k, (n, 0))[1]
    if d > 0: byproc[(k[0], comm(k[0]))] += d; bythr[(k[0], k[1], n)] += d
print(f"decode {dt:.1f}s; major faults by process:")
for k, v in byproc.most_common(8): print("  ", v, k)
print("by thread:")
for k, v in bythr.most_common(10): print("  ", v, k)
