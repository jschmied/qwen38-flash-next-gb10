#!/usr/bin/env python3
"""Trace where time goes under load on Flash-Next with PLE CPU offload.

Three candidate bottlenecks, and they are separable by process counters:
  CPU-gather bound : PLE worker utime+stime approaches wall*ncpu during the arm
  swap bound       : PLE worker major faults (majflt) climb; pswpin climbs
  GPU bound        : PLE worker mostly idle, yet decode is slow

Everything here is read from /proc and /metrics -- nothing is instrumented, so
the measurement does not perturb the timings it reports.
"""
import json, re, subprocess, sys, threading, time, urllib.request

BASE = "http://127.0.0.1:8092"
HZ = 100.0  # USER_HZ for utime/stime

def pids():
    """Read the worker PIDs from the server log.

    vLLM prefixes every line with the emitting process, e.g.
      (PleOffloadWorker pid=2981960) ...
      (Worker pid=2981775) ...
    That is exact; matching /proc/*/comm is not -- both processes inherit
    similar names and the offload worker is not always distinguishable.
    """
    ple = gpu = None
    try:
        log = open("/opt/llm/fnext-baremetal.log", errors="replace").read()
    except Exception:
        return None, None
    m = re.findall(r"\(PleOffloadWorker pid=(\d+)\)", log)
    if m: ple = int(m[-1])
    m = re.findall(r"\(Worker pid=(\d+)\)", log)
    if m: gpu = int(m[-1])
    # only report PIDs that are actually alive
    for name, pid in (("ple", ple), ("gpu", gpu)):
        if pid is not None:
            try: open(f"/proc/{pid}/stat")
            except Exception:
                if name == "ple": ple = None
                else: gpu = None
    return ple, gpu

def procstat(pid):
    if pid is None: return None
    try:
        f = open(f"/proc/{pid}/stat").read()
        # comm may contain spaces/parens -> split after the last ')'
        rest = f[f.rindex(")")+2:].split()
        return dict(minflt=int(rest[7]), majflt=int(rest[9]),
                    utime=int(rest[11]), stime=int(rest[12]))
    except Exception:
        return None

def vmstat():
    d = {}
    for line in open("/proc/vmstat"):
        k,v = line.split()
        if k in ("pswpin","pswpout","pgmajfault"): d[k]=int(v)
    return d

def metrics():
    try:
        raw = urllib.request.urlopen(BASE+"/metrics", timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for line in raw.splitlines():
        if line.startswith("#") or " " not in line: continue
        name, val = line.rsplit(" ", 1)
        try: out[name] = float(val)
        except ValueError: pass
    return out

def msum(m, prefix):
    return sum(v for k,v in m.items() if k.startswith(prefix))

PROMPTS = [
 "Explain how a B-tree index speeds up database lookups.",
 "Write a Python LRU cache class with get and put.",
 "Summarise the tradeoffs between optimistic and pessimistic locking.",
 "Describe how a copying garbage collector differs from mark-and-sweep.",
]

def one(prompt, maxtok, out, i):
    body = json.dumps({"model":"flashnext","temperature":0.6,"max_tokens":maxtok,
                       "messages":[{"role":"user","content":prompt}]}).encode()
    req = urllib.request.Request(BASE+"/v1/chat/completions", body,
                                 {"Content-Type":"application/json"})
    t0=time.time()
    try:
        d=json.load(urllib.request.urlopen(req, timeout=900))
        out[i]=(d["usage"]["completion_tokens"], d["usage"]["prompt_tokens"], time.time()-t0)
    except Exception as e:
        out[i]=(0,0,time.time()-t0)

def arm(conc, maxtok=200):
    ple, gpu = pids()
    p0, g0, v0, m0 = procstat(ple), procstat(gpu), vmstat(), metrics()
    out={}
    th=[threading.Thread(target=one,args=(PROMPTS[i%len(PROMPTS)],maxtok,out,i)) for i in range(conc)]
    t0=time.time(); [x.start() for x in th]; [x.join() for x in th]; wall=time.time()-t0
    p1, g1, v1, m1 = procstat(ple), procstat(gpu), vmstat(), metrics()

    tok = sum(o[0] for o in out.values())
    r = {"conc":conc, "wall":wall, "out_tok":tok, "tok_s":tok/wall if wall else 0}
    for name,a,b in (("ple",p0,p1),("gpu",g0,g1)):
        if a and b:
            cpu=((b["utime"]-a["utime"])+(b["stime"]-a["stime"]))/HZ
            r[f"{name}_cpu_s"]=round(cpu,2)
            r[f"{name}_cpu_pct"]=round(100*cpu/wall,1) if wall else 0
            r[f"{name}_majflt"]=b["majflt"]-a["majflt"]
            r[f"{name}_minflt"]=b["minflt"]-a["minflt"]
    r["pswpin"]=v1["pswpin"]-v0["pswpin"]
    r["pswpout"]=v1["pswpout"]-v0["pswpout"]
    # vllm-side split
    for key,pref in (("ttft_sum","vllm:time_to_first_token_seconds_sum"),
                     ("ttft_cnt","vllm:time_to_first_token_seconds_count"),
                     ("prefill_tok","vllm:prompt_tokens_total"),
                     ("decode_tok","vllm:generation_tokens_total"),
                     ("queue_sum","vllm:request_queue_time_seconds_sum"),
                     ("infer_sum","vllm:request_inference_time_seconds_sum"),
                     ("prefill_sum","vllm:request_prefill_time_seconds_sum"),
                     ("decode_sum","vllm:request_decode_time_seconds_sum")):
        r[key]=round(msum(m1,pref)-msum(m0,pref),3)
    return r

if __name__=="__main__":
    ple,gpu = pids()
    print(f"# ple_worker_pid={ple}  gpu_worker_pid={gpu}", file=sys.stderr)
    # warm the model and the page cache once; the first arm otherwise measures cold swap
    arm(1, maxtok=60)
    rows=[]
    for c in (1,2,4,8):
        r=arm(c); rows.append(r); print(json.dumps(r), flush=True)
    print("\n=== summary ===", file=sys.stderr)
    hdr=f"{'c':>2} {'tok/s':>7} {'ple cpu%':>9} {'ple majflt':>11} {'gpu cpu%':>9} {'pswpin':>8} {'ttft':>7} {'queue':>7}"
    print(hdr, file=sys.stderr)
    for r in rows:
        ttft=r["ttft_sum"]/r["ttft_cnt"] if r.get("ttft_cnt") else 0
        print(f"{r['conc']:>2} {r['tok_s']:>7.1f} {r.get('ple_cpu_pct',0):>9} "
              f"{r.get('ple_majflt',0):>11} {r.get('gpu_cpu_pct',0):>9} {r['pswpin']:>8} "
              f"{ttft:>7.2f} {r['queue_sum']:>7.2f}", file=sys.stderr)
