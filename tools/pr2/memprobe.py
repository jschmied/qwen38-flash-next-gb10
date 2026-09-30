#!/usr/bin/env python3
"""Free memory at server ready, then nvprobe (speed, greedy hashes, TTFT, replay). MemAvailable and the server's own
GPU memory view (nvidia-smi is empty on GB10, so the CUDA free/total of a fresh context) before any request. ONE json.
argv: tag"""
import json, subprocess, sys
tag = sys.argv[1]
mem = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
import torch
free, total = torch.cuda.mem_get_info()
p = subprocess.run([sys.executable, "/opt/llm/runners/nightly219/nvprobe.py", tag], capture_output=True, text=True,
                   check=True)
out = json.loads([l for l in p.stdout.splitlines() if l.strip().startswith("{")][-1])
out["mem_at_ready"] = {"memavail_gib": round(mem["MemAvailable"] / 2**20, 2), "cuda_free_gib": round(free / 2**30, 2)}
print(json.dumps(out))
