# argv: <server log> <ACC.json>. Groups ACCLOG lines by request (a new request = pos resets lower),
# prints per turn: accepted-per-step sequence, rate, and the first 60 chars of the generated text.
import re, sys, json
log, js = sys.argv[1], sys.argv[2]
rows = [(m[1], int(m[2]), int(m[3]), int(m[4])) for ln in open(log, errors="replace") for m in [re.search(r"ACCLOG req=(\S+) pos=(\d+) drafts=(\d+) acc=(\d+)", ln)] if m]
turns, cur, last = [], [], None
for req, pos, d, a in rows:
    if req != last and cur: turns.append(cur); cur = []
    cur.append((pos, d, a)); last = req
if cur: turns.append(cur)
texts = json.load(open(js))["passes"]["pass1"] if js != "-" else [""] * len(turns)
for i, t in enumerate(turns):
    d = sum(x[1] for x in t); a = sum(x[2] for x in t)
    seq = "".join(str(x[2]) for x in t)
    print(f"turn {i+1}: steps={len(t)} acc={a}/{d} ({a/max(d,1)*100:.0f}%) first-pos={t[0][0]}  seq={seq[:90]}")
    if i < len(texts): print(f"         text: {texts[i][:70]!r}")
