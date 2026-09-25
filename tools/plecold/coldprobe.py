#!/usr/bin/env python3
"""Cold-window probe on a FRESH server: PLE page-cache residency at start, then 6 different 300-token requests,
per-chunk timing + worker minflt/majflt + NVMe reads per request, and the first request in 20-step windows.
Prints ONE json (armrun contract).  argv: <arm tag>"""
import ctypes, glob, json, os, sys, time, urllib.request
URL = "http://127.0.0.1:8092"; H = {"Content-Type": "application/json", "Authorization": "Bearer sk-bench"}
PLE = "/opt/llm/models/qwen38-flash-next-mtpfp4/model-plefp8-*.safetensors"
PROMPTS = ["Explain in about 500 words how a B-tree keeps itself balanced during inserts and deletes.",
           "Write about 400 words on the history of the printing press.",
           "Describe in detail how a modern CPU branch predictor works, with examples.",
           "Write a short story (about 400 words) about a lighthouse keeper and a storm.",
           "Explain how TLS 1.3 establishes a session key, step by step.",
           "Give a detailed recipe for a sourdough bread, with timings and temperatures."]

def residency():
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]; libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    P = os.sysconf("SC_PAGE_SIZE"); tot = res = 0
    for f in glob.glob(PLE):
        sz = os.path.getsize(f); fd = os.open(f, os.O_RDONLY); a = libc.mmap(None, sz, 1, 1, fd, 0)
        n = (sz + P - 1) // P; v = (ctypes.c_ubyte * n)(); libc.mincore(a, sz, v)
        res += bytes(v).count(b"\x01") * P + bytes(v).count(b"\x81") * P; tot += sz; libc.munmap(a, sz); os.close(fd)
    return round(res / 2**30, 2), round(tot / 2**30, 2)

def workers():
    out = []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                if open(f"/proc/{p}/comm").read().strip() == "VLLM::Worker": out.append(int(p))
            except Exception: pass
    return out

def counters(pids):
    mn = mj = 0
    for p in pids:
        r = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split(); mn += int(r[7]); mj += int(r[9])
    vm = {l.split()[0]: int(l.split()[1]) for l in open("/proc/vmstat") if l.startswith(("pgmajfault", "pswpin"))}
    rd = next(int(l.split()[5]) for l in open("/proc/diskstats") if l.split()[2] == "nvme0n1")
    return {"minflt": mn, "majflt": mj, "sys_majflt": vm["pgmajfault"], "pswpin": vm["pswpin"], "nvme_rd_mb": rd * 512 / 2**20}

def decode(prompt, n):
    b = {"model": "flashnext", "temperature": 0, "max_tokens": n, "stream": True, "messages": [{"role": "user", "content": prompt}],
         "chat_template_kwargs": {"enable_thinking": False}}
    ts = []; t0 = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", json.dumps(b).encode(), H), timeout=900) as r:
        for raw in r:
            ln = raw.decode().strip()
            if ln.startswith("data:") and ln != "data: [DONE]":
                d = json.loads(ln[5:])
                if d.get("choices") and d["choices"][0].get("delta", {}).get("content"): ts.append(time.perf_counter())
    return ts, t0

def meminfo():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    return {k: round(v / 2**20, 2) for k, v in m.items() if v > 64 * 1024}

def warm():
    t0 = time.time(); buf = bytearray(64 << 20); mv = memoryview(buf)
    for f in sorted(glob.glob(PLE)):
        fd = os.open(f, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
            while os.readv(fd, [mv]) > 0: pass
        finally: os.close(fd)
    return round(time.time() - t0, 1)

res0 = residency(); pids = workers()
out = {"ple_resident_gib_at_ready": res0, "mem_at_ready": meminfo(), "workers": pids, "requests": []}
if os.environ.get("WARM") == "1" or "warm" in (sys.argv[1] if len(sys.argv) > 1 else ""):
    out["warm_s"] = warm(); out["ple_resident_gib_after_warm"] = residency(); out["mem_after_warm"] = meminfo()
for i, p in enumerate(PROMPTS):
    c0 = counters(pids); ts, t0 = decode(p, 300); c1 = counters(pids)
    gaps = [b - a for a, b in zip(ts, ts[1:])]; steps = len(gaps)
    row = {"i": i, "ttft_s": round(ts[0] - t0, 3), "steps": steps, "ms_per_step": round(1000 * sum(gaps) / steps, 2)}
    row.update({k: round((c1[k] - c0[k]) / steps, 2) for k in c0})
    if i == 0:
        row["windows_ms"] = [round(1000 * sum(gaps[j:j + 20]) / len(gaps[j:j + 20]), 1) for j in range(0, steps, 20)]
    out["requests"].append(row)
out["ple_resident_gib_at_end"] = residency(); out["mem_at_end"] = meminfo()
print(json.dumps(out))
