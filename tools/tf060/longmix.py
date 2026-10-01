#!/usr/bin/env python3
"""TensorFold 0.6.0 memory-gate probe (2026-10-01). N_LONG long prompts start at t=0, N_SHORT short ones follow every
GAP seconds; each streams its reply. Prompts: Python stdlib source with a unique first line (no cached prefix), cut to
token counts with the checkpoint's tokenizer. Samples MemAvailable and the server's major page faults every second.
Prints ONE json. argv: <tag> <tokenizer dir> [n_long long_tokens n_short short_tokens gap]"""
import json, sys, sysconfig, threading, time, urllib.request, uuid
from pathlib import Path
from transformers import AutoTokenizer

tag, tokdir = sys.argv[1], sys.argv[2]
N_LONG, LONG, N_SHORT, SHORT, GAP = (int(x) for x in (sys.argv[3:8] if len(sys.argv) > 7 else (4, 120000, 8, 2000, 15)))
URL = "http://127.0.0.1:8092/v1/chat/completions"
tok = AutoTokenizer.from_pretrained(tokdir)
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
ids = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files))
ASK = "\nSay in one sentence what the code above does."


def prompt(i: int, n: int, kind: str) -> str:
    start = (i * 7919 * 13) % max(1, len(ids) - n)               # different slices, so no shared prefix either
    return f"# {tag} {kind} {i} {uuid.uuid4().hex}\n" + tok.decode(ids[start:start + n]) + ASK


def one(kind: str, i: int, n: int, max_tokens: int, out: list) -> None:
    body = {"model": "flashnext", "messages": [{"role": "user", "content": prompt(i, n, kind)}], "max_tokens": max_tokens,
            "temperature": 1.0, "seed": 1000 + i, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    r = {"kind": kind, "i": i, "prompt_target": n, "sent": time.time() - T0}
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        first = last = None; pieces = 0
        with urllib.request.urlopen(req, timeout=7200) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                if d.get("usage"):
                    r["usage"] = d["usage"]
                for c in d.get("choices") or []:
                    if (c.get("delta") or {}).get("content"):
                        now = time.time(); first = first or now; last = now; pieces += 1
                    if c.get("finish_reason"):
                        r["finish"] = c["finish_reason"]
        ct = (r.get("usage") or {}).get("completion_tokens", 0)
        r.update(ttft=round(first - T0 - r["sent"], 2) if first else None,
                 decode_tps=round((ct - 1) / (last - first), 1) if first and last and last > first and ct > 1 else None,
                 total=round(time.time() - T0 - r["sent"], 1), pieces=pieces)
    except Exception as e:                                       # an HTTP error body names the gate's decision
        msg = e.read().decode()[:400] if hasattr(e, "read") else ""
        r["error"] = f"{type(e).__name__}: {e} {msg}".strip()
    out.append(r)


def server_pid() -> int | None:
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            except OSError:
                continue
            if "tensorfold serve" in cmd and "bash -c" not in cmd:
                return int(p.name)
    return None


samples, stop = [], threading.Event()
PID = server_pid()


def sampler() -> None:
    while not stop.is_set():
        mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo") if l.split(":")[0] in ("MemAvailable", "SwapFree")}
        majflt = None
        if PID:
            try:
                majflt = int(open(f"/proc/{PID}/stat").read().rsplit(")", 1)[1].split()[9])
            except OSError:
                pass
        samples.append((round(time.time() - T0, 1), round(mem["MemAvailable"] / 1048576, 2), majflt))
        stop.wait(1.0)


T0 = time.time()
threading.Thread(target=sampler, daemon=True).start()
results, threads = [], []
for i in range(N_LONG):
    t = threading.Thread(target=one, args=("long", i, LONG, 300, results)); t.start(); threads.append(t)
time.sleep(5)
for i in range(N_SHORT):
    t = threading.Thread(target=one, args=("short", i, SHORT, 200, results)); t.start(); threads.append(t)
    if i < N_SHORT - 1:
        time.sleep(GAP)
for t in threads:
    t.join()
stop.set()
health = None
try:
    health = json.loads(urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=10).read())
except Exception as e:
    health = {"error": str(e)}
mins = min(s[1] for s in samples) if samples else None
flt = [s[2] for s in samples if s[2] is not None]
print(json.dumps({"tag": tag, "pid": PID, "results": sorted(results, key=lambda r: (r["kind"], r["i"])),
                  "mem_available_min_gib": mins, "majflt_delta": (flt[-1] - flt[0]) if flt else None,
                  "samples": samples[::5], "health": health}))
