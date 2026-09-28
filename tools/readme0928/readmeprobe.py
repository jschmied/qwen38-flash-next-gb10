#!/usr/bin/env python3
"""README re-measure probe: run nvprobe, concprobe, agentloop_json and turnreplay against one server, in that order,
and merge their ONE-json outputs under their names (armrun contract: ONE json). argv: <tag>"""
import json, subprocess, sys
tag = sys.argv[1]; py = sys.executable; out = {}
for name, cmd in (("nv", "/opt/llm/runners/nightly219/nvprobe.py"), ("conc", "/opt/llm/runners/cg/concprobe.py"),
                  ("agent", "/opt/llm/runners/agentloop_json.py"), ("turn", "/opt/llm/runners/i54458/turnreplay.py")):
    p = subprocess.run([py, cmd, f"{tag}-{name}"], capture_output=True, text=True)
    js = [l for l in p.stdout.splitlines() if l.strip().startswith("{")]
    out[name] = json.loads(js[-1]) if js else {"error": (p.stderr or p.stdout)[-400:]}
print(json.dumps(out))
