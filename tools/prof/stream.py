"""Stream-parse a torch profiler chrome trace (multi-line events) without loading it: emit kernel rows and cpu_op rows as TSV."""
import sys, re, json
F = sys.argv[1]
ko = open("prof/kernels.tsv", "w"); co = open("prof/cpuops.tsv", "w"); ro = open("prof/runtime.tsv", "w")
ev = {}
num = re.compile(r'"([^"]+)":\s*("(?:[^"\\]|\\.)*"|[-0-9.e]+)')
def emit(e):
    c = e.get("cat")
    if c == "kernel" or c == "gpu_memcpy" or c == "gpu_memset":
        ko.write(f'{c}\t{e.get("ts")}\t{e.get("dur")}\t{e.get("External id","")}\t{e.get("correlation","")}\t{e.get("stream","")}\t{e.get("name","")}\n')
    elif c in ("cpu_op", "user_annotation", "gpu_user_annotation"):
        co.write(f'{c}\t{e.get("ts")}\t{e.get("dur")}\t{e.get("External id","")}\t{e.get("tid","")}\t{e.get("name","")}\t{e.get("Input Dims","")}\n')
    elif c == "cuda_runtime" or c == "cuda_driver":
        ro.write(f'{c}\t{e.get("ts")}\t{e.get("dur")}\t{e.get("External id","")}\t{e.get("correlation","")}\t{e.get("name","")}\n')
depth = 0
with open(F, errors="replace") as f:
    for ln in f:
        s = ln.strip()
        if s == "{" :
            ev = {}; continue
        if s.startswith("}"):
            if ev: emit(ev)
            ev = {}; continue
        for k, v in num.findall(s):
            if v.startswith('"'):
                v = v[1:-1]
            ev.setdefault(k, v)
        if '"Input Dims"' in s:
            m = re.search(r'"Input Dims":\s*(\[.*?\]\])', s)
            if m: ev["Input Dims"] = m.group(1).replace("\t", " ")
