"""Online drafter training: Kolibri-1 (TensorFold forward) prefills each sequence; the drafter learns its next states.

    python train.py MODEL_DIR PREFIX OUT_DIR [--epochs 3] [--hold 2] [--lr 3e-4] [--ce 0.1]

PREFIX.bin / PREFIX.idx.json: whole conversations (data_swe.py). Each is prefilled whole in chunks, so its states are
the ones serving sees: x[0..T) (normed, as the head reads them) and the target's own choices a[t] = argmax head(x[t]).
The drafter then takes 2,048-position windows of it, one optimizer step each. The drafter reads (x[t], embed(token[t+1])) and is trained, as EAGLE-1, on
smooth-L1 to x[t+1] plus ``ce`` x cross-entropy of head(pred) against a[t+1]. Nothing but tokens touches the disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import DraftConfig, Drafter, params  # noqa: E402

HEAD_CHUNK = 512        # rows a head matmul takes at a time (128k-vocab logits)


def target_pass(model, seq: list[int], chunk: int = 8192):
    """(states [T, D] bf16, the target's argmax [T] int64) for one whole sequence from position 0, in chunks."""

    from tensorfold.cuda import moe as shared
    from tensorfold.families.kolibri1.cuda.forward import Chain

    xs = [model.forward([Chain(0, a, seq[a:a + chunk])], prompt=True, features=True)
          for a in range(0, len(seq), chunk)]
    x = torch.cat(xs)
    picks = torch.cat([shared.router(x[i:i + HEAD_CHUNK].contiguous(), model.w.head).argmax(-1)
                       for i in range(0, x.shape[0], HEAD_CHUNK)])
    return x, picks


def conversations(prefix: str):
    """[(name, token array)] from data_swe.py's PREFIX.bin and PREFIX.idx.json."""

    idx = json.loads(Path(prefix + ".idx.json").read_text())
    data = np.memmap(prefix + ".bin", dtype=np.uint32, mode="r")
    off = idx["offsets"]
    return [(m["source"], data[off[i]:off[i + 1]]) for i, m in enumerate(idx["meta"])]


def held_out(convs, hold: int, mark: str = "kolibri"):
    """The last ``hold`` conversations whose source names ``mark`` (the target's own replies) for evaluation."""

    own = [i for i, (name, _) in enumerate(convs) if mark in name]
    out = set(own[-hold:]) if hold else set()
    return [c for i, c in enumerate(convs) if i not in out], [c for i, c in enumerate(convs) if i in out]


def head_ce(pred: torch.Tensor, head: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy of head(pred) against labels, the vocabulary in row chunks (checkpointed)."""

    from torch.utils.checkpoint import checkpoint

    def part(p, y):
        return F.cross_entropy((p.to(torch.bfloat16) @ head.T).float(), y, reduction="sum")

    total = sum(checkpoint(part, pred[i:i + HEAD_CHUNK], labels[i:i + HEAD_CHUNK], use_reentrant=False)
                for i in range(0, pred.shape[0], HEAD_CHUNK))
    return total / pred.shape[0]


def step_inputs(x: torch.Tensor, seq: torch.Tensor, picks: torch.Tensor, embed: torch.Tensor):
    """Inputs (x[t], embed(tok[t+1])), targets x[t+1] and a[t+1], for t < T - 2."""

    feats, emb = x[:-2], embed[seq[1:-1]]
    return feats, emb, x[1:-1], picks[1:-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--hold", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ce", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.0, help="uniform feature noise, +-noise")
    ap.add_argument("--log", type=int, default=50)
    ap.add_argument("--save", type=int, default=500)
    a = ap.parse_args()

    from tensorfold.families.kolibri1.cuda.forward import Model
    from tensorfold.families.kolibri1.cuda.weights import load

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, hold = held_out(conversations(a.prefix), a.hold)
    (out / "held_out.json").write_text(json.dumps([name for name, _ in hold]))
    windows_per = [max(1, (len(t) - 2 + a.window - 1) // a.window) for _, t in train]
    steps = a.epochs * sum(windows_per)
    longest = max(len(t) for _, t in train)
    w = load(a.model_dir)
    model = Model(w, longest + 64, 1)
    cfg = DraftConfig(hidden=w.config.hidden)
    dr = Drafter(cfg).cuda()
    with torch.no_grad():                                 # the target's states carry its final norm's scale (~43)
        dr.out_norm.w.copy_(w.norm.float())
    unit = float(w.norm.float().pow(2).mean().sqrt())     # L1 in units of that scale, beside the cross-entropy
    print(f"drafter: {params(dr) / 1e6:.1f}M trainable; {len(train)} conversations "
          f"({sum(len(t) for _, t in train) / 1e6:.2f}M tokens), {len(hold)} held out; {steps} steps", flush=True)
    opt = torch.optim.AdamW(dr.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.05, 1 - s / steps))
    rng = np.random.default_rng(0)
    t0, seen, step, log = time.perf_counter(), 0, 0, {"l1": 0.0, "ce": 0.0, "acc": 0.0, "n": 0}
    for epoch in range(a.epochs):
        for ci in rng.permutation(len(train)):
            name, toks = train[ci]
            seq = torch.from_numpy(np.asarray(toks, dtype=np.int64)).cuda()
            with torch.no_grad():
                x, picks = target_pass(model, seq.tolist())
            feats_all, emb_all, want_all, label_all = step_inputs(x, seq, picks, w.embed)
            seen += len(toks)
            for s0 in range(0, feats_all.shape[0], a.window):
                sl = slice(s0, s0 + a.window)
                feats, emb, want, label = feats_all[sl], emb_all[sl], want_all[sl], label_all[sl]
                if a.noise:
                    feats = feats + ((torch.rand(feats.shape, device=feats.device) * 2 - 1) * a.noise).to(feats.dtype)
                pos = torch.arange(feats.shape[0], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred, _ = dr(feats[None], emb[None], pos)
                pred = pred[0]
                l1 = F.smooth_l1_loss(pred.float() / unit, want.float() / unit)
                ce = head_ce(pred, w.head, label)
                loss = l1 + a.ce * ce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dr.parameters(), 0.5)
                opt.step()
                sched.step()
                step += 1
                with torch.no_grad():
                    k = min(256, pred.shape[0])
                    acc = ((pred[-k:].to(torch.bfloat16) @ w.head.T).argmax(-1) == label[-k:]).float().mean()
                log["l1"] += float(l1.detach())
                log["ce"] += float(ce.detach())
                log["acc"] += float(acc)
                log["n"] += 1
                if step % a.log == 0 or step == steps:
                    n, dt = log["n"], time.perf_counter() - t0
                    print(json.dumps({"epoch": epoch, "step": step, "tokens": seen, "tok_s": round(seen / dt),
                                      "l1": round(log["l1"] / n, 4), "ce": round(log["ce"] / n, 3),
                                      "top1": round(log["acc"] / n, 3), "lr": round(sched.get_last_lr()[0], 6),
                                      "h": round(dt / 3600, 2)}), flush=True)
                    log = {"l1": 0.0, "ce": 0.0, "acc": 0.0, "n": 0}
                if step % a.save == 0 or step == steps:
                    torch.save({"config": cfg.__dict__, "state": dr.state_dict(), "step": step, "tokens": seen},
                               out / "drafter.pt")


if __name__ == "__main__":
    main()
