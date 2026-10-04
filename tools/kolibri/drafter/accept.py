"""Greedy chain acceptance of a trained drafter on held-out Kolibri-1 conversations.

    python accept.py MODEL_DIR PREFIX DRAFTER.pt [--names held_out.json] [--steps 64] [--depth 6] [--turns 20]

At each assistant turn start of a held-out conversation, Kolibri decodes ``steps`` greedy tokens; along that path the
drafter, given Kolibri's state and next token, drafts ``depth`` tokens on its own. Printed: mean tokens accepted per
round at depth 1..depth (a round also yields the target's own token), and each step's conditional acceptance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import DraftConfig, Drafter  # noqa: E402
from train import conversations  # noqa: E402

CONTEXT = 1024           # drafter positions a chain attends to (teacher-forced before the start)


def turn_starts(ids: np.ndarray, start_id: int, assistant: list[int]) -> list[int]:
    """Positions just after each '<|im_start|>assistant\\n' header."""

    out, n = [], len(assistant)
    for i in np.flatnonzero(ids == start_id):
        if list(ids[i + 1:i + 1 + n]) == assistant:
            out.append(int(i + 1 + n))
    return out


@torch.no_grad()
def main() -> None:
    from tokenizers import Tokenizer

    from tensorfold.cuda import moe as shared
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model
    from tensorfold.families.kolibri1.cuda.weights import load

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("prefix")
    ap.add_argument("drafter")
    ap.add_argument("--names", default="")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--turns", type=int, default=20)
    a = ap.parse_args()
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    start_id = tok.token_to_id("<|im_start|>")
    assistant = tok.encode("assistant\n", add_special_tokens=False).ids
    names = set(json.loads(Path(a.names).read_text())) if a.names else None
    convs = [c for c in conversations(a.prefix) if names is None or c[0] in names]
    w = load(a.model_dir)
    model = Model(w, max(len(t) for _, t in convs) + a.steps + 64, 1)
    ck = torch.load(a.drafter, map_location="cuda")
    dr = Drafter(DraftConfig(**ck["config"])).cuda().to(torch.bfloat16).eval()
    dr.load_state_dict(ck["state"])
    head, embed = w.head, w.embed
    matched = []                                  # per round: drafts that matched, out of depth
    for name, toks in convs:
        ids = np.asarray(toks, dtype=np.int64)
        starts = turn_starts(ids, start_id, assistant)[:a.turns]
        feats, done = [], 0
        for s in starts:
            for p in range(done, s, 8192):            # the conversation's prompt rows up to this turn
                feats.append(model.forward([Chain(0, p, ids[p:min(s, p + 8192)].tolist())], prompt=True,
                                           features=True))
            done = s
            prev = torch.cat(feats)[-CONTEXT:] if feats else None
            # greedy path from s: the target's states and tokens (decode rows, as serving makes them)
            x0 = model.forward([Chain(0, s - 1, [int(ids[s - 1])])], prompt=False, features=True)
            path, pf = [], []
            x = x0
            for i in range(a.steps):
                t = int(shared.router(x, head).argmax(-1)[0])
                path.append(t)
                pf.append(x)
                x = model.forward([Chain(0, s + i, [t])], prompt=False, features=True)
            pf = torch.cat(pf)                          # pf[i]: the state that chose path[i]
            # drafter context: teacher-forced over the prompt rows before the turn, then the path
            ctx_f = torch.cat([prev[:-1], pf]) if prev is not None and len(prev) > 1 else pf
            ctx_t = np.concatenate([ids[s - len(ctx_f) + len(pf):s], path])[:len(ctx_f)]
            _, kv = dr(ctx_f[None], embed[torch.as_tensor(ctx_t, device="cuda")][None],
                       torch.arange(len(ctx_f), device="cuda"))
            off = len(ctx_f) - len(pf)
            for i in range(len(pf) - a.depth - 1):
                past = (kv[0][:, :, :off + i], kv[1][:, :, :off + i])
                f, nxt, n = pf[i:i + 1], path[i], 0
                for k in range(a.depth):
                    out, step_kv = dr(f[None], embed[torch.as_tensor([nxt], device="cuda")][None],
                                      torch.tensor([off + i + k], device="cuda"), past)
                    past = step_kv
                    f = out[0]
                    nxt = int((f.to(torch.bfloat16) @ head.T).argmax(-1)[0])
                    if nxt != path[i + 1 + k]:
                        break
                    n += 1
                matched.append(n)
        print(f"{name}: {len(starts)} turns, {len(matched)} rounds so far", flush=True)
    m = np.asarray(matched)
    res = {"rounds": int(len(m)), "accepted_at_depth": {d: round(float(np.minimum(m, d).mean()), 3)
                                                        for d in range(1, a.depth + 1)},
           "step_acceptance": [round(float((m >= k).sum() / max(1, (m >= k - 1).sum())), 3)
                               for k in range(1, a.depth + 1)]}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
