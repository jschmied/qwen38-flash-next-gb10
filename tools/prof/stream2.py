"""Stream-parse a torch profiler trace; kernels with grid/block. argv: trace out_dir"""
import sys, re, os
F, O = sys.argv[1], sys.argv[2]; os.makedirs(O, exist_ok=True)
ko = open(f"{O}/kernels2.tsv", "w")
ev = {}; kv = re.compile(r'"([^"]+)":\s*("(?:[^"\\]|\\.)*"|[-0-9.e]+|\[[^\]]*\])')
for ln in open(F, errors="replace"):
    s = ln.strip()
    if s == "{": ev = {}; continue
    if s.startswith("}"):
        if ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            ko.write("\t".join(str(ev.get(k, "")) for k in ("cat", "ts", "dur", "stream", "grid", "block", "name")) + "\n")
        ev = {}; continue
    for k, v in kv.findall(s):
        ev.setdefault(k, v.strip('"').replace(" ", "") if not v.startswith('"') else v[1:-1])
