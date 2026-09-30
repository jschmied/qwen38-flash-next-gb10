#!/usr/bin/env python3
"""FNHEAD4 screen, one server start: nvprobe (speed, hashes, TTFT, replay; its keys at the top level for armrun's
summary), then tfprobe (teacher-forced NLL / top-1 on reference text) under "tf", then evalprobe (GSM8K + HumanEval
greedy c=16; generations in results/evalq-<tag>.jsonl for evalscore.py) under "eval". ONE json. argv: tag"""
import json, subprocess, sys
tag = sys.argv[1]


def run(path):
    p = subprocess.run([sys.executable, path, tag], capture_output=True, text=True, check=True)
    return json.loads([l for l in p.stdout.splitlines() if l.strip().startswith("{")][-1])


out = run("/opt/llm/runners/nightly219/nvprobe.py")
out["tf"] = run("/opt/llm/runners/head4/tfprobe.py")
out["eval"] = run("/opt/llm/runners/evalq/evalprobe.py")
print(json.dumps(out))
