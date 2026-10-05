"""Score drafter checkpoints on recordings, optionally only those finished after a time (unseen by a training run).

    python eval_rec.py MODEL_DIR --data DIR [--after 'YYYY-MM-DD HH:MM'] [--contains TEXT] [--excludes TEXT]
        --ckpt A.pt [B.pt ...] [--layers 1]

A checkpoint with a different tap count (run 3: one tap) is loaded the way train_rec.py warm-starts it: the fuse passes
the last tap (the head's state) through.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model_mt import DraftConfig, Drafter  # noqa: E402
from model_mt import load as load_drafter  # noqa: E402
from train_rec import head_weights, held_score, recordings  # noqa: E402


def drafter(path: str, layers: int, taps: int, dims: int) -> Drafter:
    ck = torch.load(path, map_location="cuda")
    old = load_drafter(ck, layers)
    if old.cfg.taps == taps:
        return old.cuda().eval()
    dr = Drafter(DraftConfig(**{**old.cfg.__dict__, "taps": taps}))
    dr.load_state_dict(old.state_dict(), strict=False)
    with torch.no_grad():
        dr.fuse.weight.zero_()
        dr.fuse.weight[:, -dims:] = torch.eye(dims)
    return dr.cuda().eval()


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--after", default="")
    ap.add_argument("--contains", default="", help="only conversations whose first 2,000 tokens contain this text (a|b: either)")
    ap.add_argument("--excludes", default="", help="only conversations whose first 2,000 tokens lack this text (a|b: all of them)")
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--rollout", type=int, default=3)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--draft-vocab", default="", help="JSON token ids: the drafter proposes only these (as served)")
    a = ap.parse_args()
    runs = recordings(a.data)
    if a.after:
        t = time.mktime(time.strptime(a.after, "%Y-%m-%d %H:%M"))
        runs = [r for r in runs if Path(r["name"] + ".json").stat().st_mtime > t]
    if a.contains or a.excludes:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
        text = {r["name"]: tok.decode([int(t) for t in r["tok"][:2000]]) for r in runs}
        has = lambda name, alts: any(x in text[name] for x in alts.split("|"))
        runs = [r for r in runs if (not a.contains or has(r["name"], a.contains))
                and (not a.excludes or not has(r["name"], a.excludes))]
    if not runs:
        raise SystemExit("no recordings match")
    embed, head = head_weights(a.model_dir)
    dims, taps = embed.shape[1], len(runs[0]["taps"])
    print(f"{len(runs)} runs, {sum(r['n'] for r in runs)} rows", flush=True)
    vocab = None
    if a.draft_vocab:
        vocab = torch.tensor(json.loads(Path(a.draft_vocab).read_text()), device=head.device)
        head = head[vocab].contiguous()
    for path in a.ckpt:
        dr = drafter(path, a.layers, taps, dims)
        print(json.dumps({"ckpt": path, "vocab": len(vocab) if vocab is not None else "full",
                          **held_score(dr, runs, embed, head, a.rollout, a.window, dims, vocab)}), flush=True)


if __name__ == "__main__":
    main()
