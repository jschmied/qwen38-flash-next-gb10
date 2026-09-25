#!/usr/bin/env python3
"""decprobe twice (second, warm pass kept) + divprobe; merge into one json (armrun contract). argv: <arm tag>"""
import json, subprocess, sys
py = sys.executable; tag = sys.argv[1]
subprocess.run([py, "/opt/llm/runners/fullcg/decprobe.py", tag], capture_output=True, text=True)
d = subprocess.run([py, "/opt/llm/runners/fullcg/decprobe.py", tag], capture_output=True, text=True)
v = subprocess.run([py, "/opt/llm/runners/gdncs/divprobe.py", tag], capture_output=True, text=True)
out = json.loads(d.stdout.strip().splitlines()[-1]); out["div"] = json.loads(v.stdout.strip().splitlines()[-1])
print(json.dumps(out))
