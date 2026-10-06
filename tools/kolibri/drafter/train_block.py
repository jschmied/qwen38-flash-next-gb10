"""Train the block drafter (model_block.py) on Kolibri-1's prefill states, as train_mt.py does for the chain drafter.

    python train_block.py MODEL_DIR OUT_DIR --data PREFIX [PREFIX ...] --draft-vocab V.json [--taps 44,47,49]
        [--block 4] [--layers 4] [--ffn 6144] [--anchors 512] [--lr 4e-4] [--max-tokens N] [--init drafter.pt]

Each conversation (data_glm.py / data_text.py format) is prefilled whole; per 2,048-row window ``anchors`` rows are
drawn, and each anchor's block learns KL(Kolibri || drafter) over the draft vocabulary at its ``block`` positions,
weighted 0.8^j. Every conversation once, in one shuffled stream; checkpoints every ``--save`` steps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import train_mt  # noqa: E402
from model_block import BlockConfig, BlockDrafter, load  # noqa: E402
from model_mt import params  # noqa: E402


def save(dr: BlockDrafter, step: int, seen: int, out: Path) -> None:
    """bf16 weights (half the disk; every save is kept by keepckpt.sh), written whole before the rename."""

    state = {k: v.to(torch.bfloat16) for k, v in dr.state_dict().items()}
    torch.save({"config": dr.cfg.__dict__, "state": state, "step": step, "tokens": seen, "taps": train_mt.TAPS,
                "kind": "block"}, out / "drafter.pt.tmp")
    (out / "drafter.pt.tmp").rename(out / "drafter.pt")


def main() -> None:
    from tensorfold.families.kolibri1.cuda.forward import Model
    from tensorfold.families.kolibri1.cuda.weights import load as load_kolibri

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--draft-vocab", required=True)
    ap.add_argument("--taps", default="44,47,49")
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--ffn", type=int, default=6144)
    ap.add_argument("--anchors", type=int, default=512)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr-end", type=float, default=0.0, help="cosine from --lr down to this by tokens seen (0: constant)")
    ap.add_argument("--lr-span", type=int, default=0, help="tokens the cosine spans (0: the whole stream)")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--init", default="")
    ap.add_argument("--decay", default="0.8", help="position weights decay^j; 'a:b:N' eases from a to b over N tokens")
    ap.add_argument("--skip-tokens", type=int, default=0, help="skip the stream's first N tokens (a resumed run)")
    ap.add_argument("--log", type=int, default=50)
    ap.add_argument("--save", type=int, default=500)
    a = ap.parse_args()
    train_mt.TAPS[:] = [int(v) for v in a.taps.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    convs = [c for p in a.data for c in train_mt.conversations(p)]
    w = load_kolibri(a.model_dir)
    model = Model(w, max(len(t) for _, t in convs) + 64, 1)
    vocab = torch.tensor(json.loads(Path(a.draft_vocab).read_text()), device="cuda")
    head = w.head[vocab].contiguous()
    cfg = BlockConfig(hidden=w.config.hidden, layers=a.layers, ffn=a.ffn, taps=len(train_mt.TAPS), block=a.block)
    dr = (load(torch.load(a.init, map_location="cuda")) if a.init else BlockDrafter(cfg)).cuda()
    cfg = dr.cfg
    if not a.init:
        with torch.no_grad():                             # the target's states carry its final norm's scale (~43)
            dr.out_norm.w.copy_(w.norm.float())
    opt = torch.optim.AdamW(dr.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"block drafter: {params(dr) / 1e6:.1f}M trainable, block {cfg.block}, {cfg.layers} layers, taps "
          f"{train_mt.TAPS}; {len(convs)} conversations ({sum(len(t) for _, t in convs) / 1e6:.2f}M tokens)", flush=True)
    d0, d1, span = (float(v) for v in a.decay.split(":")) if ":" in a.decay else (float(a.decay),) * 2 + (1.0,)
    jj = torch.arange(cfg.block, device="cuda", dtype=torch.float32)
    rng = np.random.default_rng(0)
    t0, step, seen, log = time.perf_counter(), 0, 0, {"kl": 0.0, "top1": 0.0, "n": 0}
    skipped, total = 0, sum(len(t) for _, t in convs)
    for ci in rng.permutation(len(convs)):
        name, toks = convs[ci]
        if skipped < a.skip_tokens:                       # same seed, same order: the resumed run's first tokens
            skipped += len(toks)
            seen += len(toks)
            continue
        if a.max_tokens and seen > a.max_tokens:
            break
        seq = torch.from_numpy(np.asarray(toks, dtype=np.int64)).cuda()
        with torch.no_grad():
            x, taps, _ = train_mt.target_pass(model, seq.tolist())
        seen += len(toks)
        n = x.shape[0]
        for s0 in range(0, n, a.window):
            s1 = min(s0 + a.window, n)
            last = s1 - 1 - cfg.block                    # an anchor t needs rows t+1..t+block in the window
            if last < s0:
                continue
            cand = np.arange(s0, last + 1)
            pick = np.sort(rng.choice(cand, size=min(a.anchors, len(cand)), replace=False))
            anchors = torch.from_numpy(pick).cuda()
            j = torch.arange(cfg.block, device="cuda")
            tgt_rows = anchors[:, None] + 1 + j[None]              # Kolibri's state whose head guesses draft j
            with torch.no_grad():
                logp = F.log_softmax((x[tgt_rows].reshape(-1, x.shape[1]).to(torch.bfloat16) @ head.T).float(), -1)
            lr = a.lr
            if a.lr_end:
                frac = min(1.0, seen / (a.lr_span or total))
                lr = a.lr_end + (a.lr - a.lr_end) * 0.5 * (1 + math.cos(math.pi * frac))
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / a.warmup)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                states = dr(taps[s0:s1], w.embed[seq[anchors + 1]], anchors - s0)
            logq = F.log_softmax((states.reshape(-1, states.shape[-1]).to(torch.bfloat16) @ head.T).float(), -1)
            kl = (logp.exp() * (logp - logq)).sum(-1).view(len(pick), cfg.block)
            weights = (d0 + (d1 - d0) * min(1.0, seen / span)) ** jj
            loss = (kl * weights).sum(-1).mean() / weights.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dr.parameters(), 0.5)
            opt.step()
            step += 1
            with torch.no_grad():
                log["top1"] += float((logq.view(len(pick), cfg.block, -1)[:, 0].argmax(-1)
                                      == logp.view(len(pick), cfg.block, -1)[:, 0].argmax(-1)).float().mean())
            log["kl"] += float(loss.detach())
            log["n"] += 1
            if step % a.log == 0:
                k, dt = log["n"], time.perf_counter() - t0
                print(json.dumps({"step": step, "tokens": seen, "tok_s": round((seen - skipped) / dt), "kl": round(log["kl"] / k, 4),
                                  "top1": round(log["top1"] / k, 3),
                                  "decay": round(float(weights[1]), 3), "lr": f"{opt.param_groups[0]['lr']:.2e}", "h": round(dt / 3600, 2)}), flush=True)
                log = {"kl": 0.0, "top1": 0.0, "n": 0}
            if step % a.save == 0:
                save(dr, step, seen, out)
    save(dr, step, seen, out)
    print(json.dumps({"done": step, "tokens": seen}), flush=True)


if __name__ == "__main__":
    main()
