#!/usr/bin/env python3
"""Item 3: decprobe (c=1/c=4 + agent loop, warm pass kept) + concprobe (c=4/8/16). ONE json. argv: <tag>"""
import json, subprocess, sys
py = sys.executable; tag = sys.argv[1]
def run(p):
    r = subprocess.run([py, p, tag], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])
run("/opt/llm/runners/fullcg/decprobe.py")
out = run("/opt/llm/runners/fullcg/decprobe.py")
out["conc"] = run("/opt/llm/runners/cg/concprobe.py")
print(json.dumps(out))
