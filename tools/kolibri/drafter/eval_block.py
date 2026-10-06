"""Score block drafters on recordings (train_rec.py's format), counted as held_score counts the chain drafter.

    python eval_block.py MODEL_DIR --data DIR --draft-vocab V.json --ckpt A.pt [...] [--after 'YYYY-MM-DD HH:MM']
        [--contains TEXT] [--excludes TEXT]

Per 2,048-row window, every row t whose first draft (the token at t+2) Kolibri generated is an anchor; the block's
drafts j = 0.. are compared with Kolibri's top-1 at row t+1+j and counted while they match, and Kolibri generated
the drafted token, in a row. Printed: accepted
drafts per round within the first 3 (comparable with the chain drafter at depth 3) and within the whole block, and each
position's top-1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_block import load  # noqa: E402
from train_rec import head_weights, recordings  # noqa: E402

WINDOW = 2048
CHUNK = 512


@torch.no_grad()
def score(dr, runs, embed, head, vocab) -> dict:
    b = dr.cfg.block
    n, acc3, accb, hits = 0, 0.0, 0.0, [0] * b
    for run in runs:
        for s0 in range(0, run["n"], WINDOW):
            s1 = min(s0 + WINDOW, run["n"])
            kind = np.asarray(run["kind"][s0:s1 + 1], dtype=np.bool_)
            rows = np.arange(s0, s1 - 1 - b)
            rows = rows[kind[rows - s0 + 2] == 1] if len(rows) else rows
            if not len(rows):
                continue
            feats = torch.from_numpy(np.asarray(run["st"][s0:s1])).cuda().view(torch.bfloat16)
            tok = torch.from_numpy(np.asarray(run["tok"][s0:s1], dtype=np.int64)).cuda()
            top1 = torch.from_numpy(np.asarray(run["tki"][s0:s1, 0], dtype=np.int64)).cuda()
            gen = torch.from_numpy(kind).cuda()
            for c in range(0, len(rows), CHUNK):
                anchors = torch.from_numpy(rows[c:c + CHUNK] - s0).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    st = dr(feats, embed[tok[anchors + 1]], anchors)
                got = vocab[(st.reshape(-1, st.shape[-1]).to(torch.bfloat16) @ head.T).argmax(-1)].view(len(anchors), b)
                want = top1[anchors[:, None] + 1 + torch.arange(b, device="cuda")[None]]
                ok = (got == want) & gen[anchors[:, None] + 2 + torch.arange(b, device="cuda")[None]]
                chain = ok.long().cumprod(-1)
                acc3 += float(chain[:, :3].sum())
                accb += float(chain.sum())
                for j in range(b):
                    hits[j] += int(ok[:, j].sum())
                n += len(anchors)
    return {"accepted_3": round(acc3 / max(1, n), 4), f"accepted_{b}": round(accb / max(1, n), 4),
            "top1_steps": [round(h / max(1, n), 4) for h in hits], "chains": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--draft-vocab", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--after", default="")
    ap.add_argument("--contains", default="")
    ap.add_argument("--excludes", default="")
    a = ap.parse_args()
    runs = recordings(a.data)
    if a.after:
        t = time.mktime(time.strptime(a.after, "%Y-%m-%d %H:%M"))
        runs = [r for r in runs if Path(r["name"] + ".json").stat().st_mtime > t]
    if a.contains or a.excludes:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
        text = {r["name"]: tk.decode([int(t) for t in r["tok"][:2000]]) for r in runs}
        has = lambda name, alts: any(x in text[name] for x in alts.split("|"))  # noqa: E731
        runs = [r for r in runs if (not a.contains or has(r["name"], a.contains))
                and (not a.excludes or not has(r["name"], a.excludes))]
    embed, head = head_weights(a.model_dir)
    vocab = torch.tensor(json.loads(Path(a.draft_vocab).read_text()), device="cuda")
    head = head[vocab].contiguous()
    print(f"{len(runs)} runs, {sum(r['n'] for r in runs)} rows", flush=True)
    for path in a.ckpt:
        dr = load(torch.load(path, map_location="cuda")).cuda().eval()
        print(json.dumps({"ckpt": path, **score(dr, runs, embed, head, vocab)}), flush=True)


if __name__ == "__main__":
    main()
