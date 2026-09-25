#!/usr/bin/env python3
"""Phase 2 (prefix caching on): decprobe twice (pass 2 = prefix-cache hits on the same prompts), keep both passes'
hashes so cache-hit replay can be checked for self-consistency, then divprobe. argv: <arm tag>"""
import json, subprocess, sys
py = sys.executable; tag = sys.argv[1]
def run(p):
    r = subprocess.run([py, p, tag], capture_output=True, text=True)
    return json.loads(r.stdout.strip().splitlines()[-1])
p1 = run("/opt/llm/runners/fullcg/decprobe.py")
out = run("/opt/llm/runners/fullcg/decprobe.py")
out["pass1"] = {k: {"shas": p1[k].get("shas"), "ms": p1[k].get("decode_ms_per_tok", p1[k].get("agg_tok_s"))}
                for k in ("c1", "c4") if k in p1}
out["cache_replay_match"] = {k: [a == b for a, b in zip(p1[k]["shas"], out[k]["shas"])]
                             for k in ("c1", "c4") if k in p1 and k in out}
out["div"] = run("/opt/llm/runners/gdncs/divprobe.py")
print(json.dumps(out))
