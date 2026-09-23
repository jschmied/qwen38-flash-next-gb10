# Generation equivalence (finding 226, s6), v3 after review 2 (2026-09-23).
# Every request must succeed (futures .result()); all keys required; mixed-batch overlap PROVEN from streamed
# token timestamps. Per server: A-set (8 sequential prompts, c=1, MTP), then the mixed batch 3x (same decode
# prompts, per-rep salted prefills so the prefix cache cannot skip them), then each decode prompt alone.
# Decode hash equal across reps and equal to solo = batch-invariant on this server.
import json, sys, time, hashlib, urllib.request
from concurrent.futures import ThreadPoolExecutor
URL = "http://127.0.0.1:8092/v1/chat/completions"; KEY = "sk-bench"
label = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
corpus = open("/opt/llm/runners/corpus_frozen.txt").read()
DEC = [f"[eq-B-dec-{j}] Write a detailed essay about memory hierarchies in GPUs, part {j}." for j in range(2)]

def ask(text, max_tokens):
    b = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": max_tokens, "stream": True,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": text}],
                    "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(URL, b, {"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    t_start = time.time(); first = last = None; parts = []; ntok = None
    with urllib.request.urlopen(r, timeout=900) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            for c in d.get("choices", []):
                piece = (c.get("delta") or {}).get("content")
                if piece:
                    now = time.time(); first = first or now; last = now; parts.append(piece)
    text = "".join(parts)
    if ntok is None or not text:
        raise RuntimeError(f"empty or usage-less completion (ntok={ntok})")
    return {"hash": hashlib.sha256(text.encode()).hexdigest()[:12], "ntok": ntok,
            "t_start": t_start, "t_first": first, "t_last": last}

out = {"label": label}
off = 1_000_000
out["A"] = []
for i in range(8):
    n = 6_000 + 4_000 * i
    out["A"].append(ask(f"[eq-A-{i}]\n" + corpus[off:off + n] + "\nExplain what the code above does, step by step.", 200)["hash"])
    off += n
out["B"] = []
for rep in range(REPS):
    with ThreadPoolExecutor(4) as ex:
        fd = [ex.submit(ask, DEC[j], 400) for j in range(2)]
        time.sleep(1.5)
        fp = [ex.submit(ask, f"[eq-B-pre-{j}-rep{rep}]\n" + corpus[off + j * 70_000: off + (j + 1) * 70_000]
                        + "\nSummarise the code above.", 64) for j in range(2)]
        d = [f.result() for f in fd]; p = [f.result() for f in fp]
    ov = all(x["t_first"] < y["t_start"] and x["t_last"] > y["t_first"] for x in d for y in p)
    out["B"].append({"dec": [x["hash"] for x in d], "pre": [y["hash"] for y in p], "overlap": ov})
out["solo"] = [ask(DEC[j], 400)["hash"] for j in range(2)]
assert len(out["A"]) == 8 and len(out["B"]) == REPS and len(out["solo"]) == 2
out["overlap_all"] = all(b["overlap"] for b in out["B"])
out["dec_stable_across_reps"] = all(b["dec"] == out["B"][0]["dec"] for b in out["B"])
out["dec_equals_solo"] = [all(b["dec"][j] == out["solo"][j] for b in out["B"]) for j in range(2)]
print(json.dumps(out))
if not out["overlap_all"]:
    raise SystemExit("EQUIV VOID: a mixed batch did not overlap")
