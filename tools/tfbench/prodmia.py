#!/usr/bin/env python3
"""One server, two probes: our nvprobe (TTFT, decode, c=4, replay, hashes) and MiaAI's two decode prompts (miaprobe),
merged into ONE json line (nvprobe's keys at the top level for armrun's summary, MiaAI's under "mia"). argv: tag"""
import json, subprocess, sys
tag = sys.argv[1]
nv = subprocess.run([sys.executable, "/opt/llm/runners/nightly219/nvprobe.py", tag], capture_output=True, text=True,
                    check=True).stdout.strip().splitlines()[-1]
mia = subprocess.run([sys.executable, "/opt/llm/runners/tfbench/miaprobe.py", tag], capture_output=True, text=True,
                     check=True).stdout.strip().splitlines()[-1]
out = json.loads(nv)
out["mia"] = json.loads(mia)
print(json.dumps(out))
