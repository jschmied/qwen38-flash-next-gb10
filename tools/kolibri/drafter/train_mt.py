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
from model_mt import DraftConfig, Drafter, params  # noqa: E402
from model_mt import load as load_drafter  # noqa: E402

HEAD_CHUNK = 512        # rows a head matmul takes at a time (128k-vocab logits)


TAPS: list[int] = [12, 25, 49]   # Kolibri-1's layers whose outputs the drafter reads (49: the head's own state)


def target_pass(model, seq: list[int], chunk: int = 8192):
    """(states [T, D], the tapped layers' states [T, taps * D], the target's argmax [T]) for one sequence, in chunks."""

    from tensorfold.cuda import moe as shared
    from tensorfold.families.kolibri1.cuda.forward import Chain

    xs, ts = [], []
    for a in range(0, len(seq), chunk):
        x, states = model.forward([Chain(0, a, seq[a:a + chunk])], prompt=True, features=True, taps=TAPS)
        xs.append(x)
        ts.append(torch.cat(states, -1))
    x, taps = torch.cat(xs), torch.cat(ts)
    picks = torch.cat([shared.router(x[i:i + HEAD_CHUNK].contiguous(), model.w.head).argmax(-1)
                       for i in range(0, x.shape[0], HEAD_CHUNK)])
    return x, taps, picks


def grow_ffn(old: Drafter, width: int) -> Drafter:
    """The drafter with each block's FFN widened to ``width``: old units copied, new ones with zero output weights, so the
    function is unchanged at start; the new input rows keep Drafter's fresh init."""

    new = Drafter(DraftConfig(**{**old.cfg.__dict__, "ffn": width})).to(next(old.parameters()).device)
    keep = {k: v for k, v in old.state_dict().items() if not k.endswith(("gate.weight", "up.weight", "down.weight"))}
    missing, unexpected = new.load_state_dict(keep, strict=False)
    assert not unexpected and all(m.endswith(("gate.weight", "up.weight", "down.weight")) for m in missing), missing
    f = old.cfg.ffn
    with torch.no_grad():
        for a, b in zip(old.blocks, new.blocks):
            b.gate.weight[:f].copy_(a.gate.weight)
            b.up.weight[:f].copy_(a.up.weight)
            b.down.weight.zero_()
            b.down.weight[:, :f].copy_(a.down.weight)
    return new


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


def plain_ce(pred: torch.Tensor, head: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """head_ce's loss in one matmul (logits kept for the backward instead of recomputed)."""

    return F.cross_entropy((pred.to(torch.bfloat16) @ head.T).float(), labels)


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


def chain_batch(x, taps, seq, picks, embed, s0: int, window: int, steps: int):
    """Chains t in [s0, s0 + window) with all ``steps`` targets: feats, the step inputs, target states and choices."""

    t1 = min(s0 + window, x.shape[0] - steps)
    if t1 <= s0:
        return None
    rows = torch.arange(s0, t1, device=x.device)
    embs = [embed[seq[rows + 1 + j]] for j in range(steps)]
    wants = [x[rows + 1 + j] for j in range(steps)]
    labels = [picks[rows + 1 + j] for j in range(steps)]
    return taps[rows], embs, wants, labels, rows


def assistant_mask(ids: np.ndarray, start_id: int, header: list[int], end_id: int) -> np.ndarray:
    """True on tokens inside assistant turns (after '<|im_start|>assistant\\n', through '<|im_end|>')."""

    mask, inside, n = np.zeros(len(ids), dtype=bool), False, len(header)
    i = 0
    while i < len(ids):
        if ids[i] == start_id and list(ids[i + 1:i + 1 + n]) == header:
            inside, i = True, i + 1 + n
            continue
        if inside:
            mask[i] = True
            if ids[i] == end_id:
                inside = False
        i += 1
    return mask


class HeldOut:
    """The held-out conversations' target states, computed once; ``score`` is the drafter's step-1 top-1 there."""

    def __init__(self, model, convs, model_dir: str, window: int, steps: int = 1) -> None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
        header = tok.encode("assistant\n", add_special_tokens=False).ids
        self.items, self.window, self.steps = [], window, steps
        for _, toks in convs:
            ids = np.asarray(toks, dtype=np.int64)
            seq = torch.from_numpy(ids).cuda()
            with torch.no_grad():
                x, taps, picks = target_pass(model, ids.tolist())
            mask = torch.from_numpy(assistant_mask(ids, start, header, end)).cuda()
            self.items.append((x, taps, seq, picks, mask))

    @torch.no_grad()
    def score(self, dr, embed, head) -> dict:
        """Per-step top-1 on chains whose first drafted token is the assistant's, and mean drafts accepted in a row."""

        n, accepted = 0, 0.0
        hits = [0] * self.steps
        for x, taps, seq, picks, mask in self.items:
            for s0 in range(0, x.shape[0], self.window):
                b = chain_batch(x, taps, seq, picks, embed, s0, self.window, self.steps)
                if b is None:
                    continue
                feats, embs, _, labels, rows = b
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = dr.rollout(feats, embs)
                keep = mask[(rows + 2).clamp(max=mask.shape[0] - 1)]   # chain t drafts the token at t + 2 first
                run = torch.ones_like(keep)
                for j, (o, lab) in enumerate(zip(outs, labels)):
                    got = torch.cat([(o[i:i + HEAD_CHUNK].to(torch.bfloat16) @ head.T).argmax(-1)
                                     for i in range(0, o.shape[0], HEAD_CHUNK)])
                    ok = (got == lab) & keep
                    hits[j] += int(ok.sum())
                    run = run & ok
                    accepted += float(run.sum())
                n += int(keep.sum())
        return {"accepted": round(accepted / max(1, n), 4), "top1_steps": [round(h / max(1, n), 4) for h in hits],
                "chains": n}


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
    ap.add_argument("--extra", action="append", default=[], help="more PREFIXes trained on (never held out)")
    ap.add_argument("--swe-passes", type=int, default=0, help="with --extra: passes over PREFIX beside one of them")
    ap.add_argument("--init", default="", help="a drafter.pt to continue from")
    ap.add_argument("--eval", type=int, default=250, help="steps between held-out checks (0: none)")
    ap.add_argument("--rollout", type=int, default=1, help="draft steps trained per chain (training-time test)")
    ap.add_argument("--layers", type=int, default=0, help="grow the --init drafter to this many layers (0: keep)")
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--taps", default=",".join(map(str, TAPS)), help="target layers the drafter reads")
    ap.add_argument("--plateau", type=float, default=0.005, help="held-out gain over 3 checks that still counts")
    ap.add_argument("--decay", type=float, default=0.1, help="decay phase, as a share of the steps before it")
    ap.add_argument("--loss", choices=("l1ce", "kl", "kll1"), default="l1ce",
                    help="l1ce: L1 to the next state + --ce x CE on Kolibri's choice (EAGLE-1); kl: KL to Kolibri's "
                         "distribution only (EAGLE-3); kll1: KL + --l1w x L1")
    ap.add_argument("--l1w", type=float, default=1.0, help="weight of the state L1 (l1ce, kll1)")
    ap.add_argument("--draft-vocab", default="", help="JSON token ids: losses (and top-1) over this slice only")
    ap.add_argument("--max-tokens", type=int, default=0, help="stop after this many training tokens (0: all)")
    ap.add_argument("--grow-ffn", type=int, default=0, help="widen the --init drafter's FFN to this width, loss-free")
    ap.add_argument("--plain-ce", action="store_true",
                    help="cross-entropy without chunked recompute (same loss, ~30 %% faster step, ~5 GB more memory)")
    a = ap.parse_args()
    TAPS[:] = [int(v) for v in a.taps.split(",")]

    from tensorfold.families.kolibri1.cuda.forward import Model
    from tensorfold.families.kolibri1.cuda.weights import load

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train, hold = held_out(conversations(a.prefix), a.hold)
    (out / "held_out.json").write_text(json.dumps([name for name, _ in hold]))
    if a.extra:                     # fresh text once, the first PREFIX ``swe_passes`` times, one shuffled stream
        train = train * a.swe_passes + [c for p in a.extra for c in conversations(p)]
        a.epochs = 1
    windows_per = [max(1, (len(t) - 2 + a.window - 1) // a.window) for _, t in train]
    steps = a.epochs * sum(windows_per)
    longest = max(len(t) for _, t in train + hold)
    w = load(a.model_dir)
    model = Model(w, longest + 64, 1)
    if a.init:
        old = load_drafter(torch.load(a.init, map_location="cuda"), a.layers or None)
        if old.cfg.taps == len(TAPS):
            dr = old.cuda()
        else:                                         # a drafter on fewer taps: the fuse passes the last one (the head's state)
            dr = Drafter(DraftConfig(**{**old.cfg.__dict__, "taps": len(TAPS)}))
            missing, unexpected = dr.load_state_dict(old.state_dict(), strict=False)
            assert not unexpected and all(m.startswith("fuse") for m in missing), missing
            with torch.no_grad():
                dr.fuse.weight.zero_()
                dr.fuse.weight[:, -w.config.hidden:] = torch.eye(w.config.hidden)
            dr = dr.cuda()
    else:
        dr = Drafter(DraftConfig(hidden=w.config.hidden, layers=a.layers or 1, ffn=a.ffn, taps=len(TAPS))).cuda()
    if a.init and a.grow_ffn > dr.cfg.ffn:
        dr = grow_ffn(dr, a.grow_ffn).cuda()
        print(f"FFN grown to {a.grow_ffn}: {params(dr) / 1e6:.1f}M trainable", flush=True)
    cfg = dr.cfg
    if not a.init:
        with torch.no_grad():                             # the target's states carry its final norm's scale (~43)
            dr.out_norm.w.copy_(w.norm.float())
    unit = float(w.norm.float().pow(2).mean().sqrt())     # L1 in units of that scale, beside the cross-entropy
    print(f"drafter: {params(dr) / 1e6:.1f}M trainable; {len(train)} conversations "
          f"({sum(len(t) for _, t in train) / 1e6:.2f}M tokens), {len(hold)} held out; {steps} steps", flush=True)
    opt = torch.optim.AdamW(dr.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    held = HeldOut(model, hold, a.model_dir, a.window, max(1, a.rollout)) if a.eval and hold else None
    if a.draft_vocab:                               # the drafter's vocabulary slice: losses over it, out-of-slice CE ignored
        vocab = torch.tensor(json.loads(Path(a.draft_vocab).read_text()), device="cuda")
        head = w.head[vocab].contiguous()
        remap = torch.full((w.head.shape[0],), -100, dtype=torch.long, device="cuda")
        remap[vocab] = torch.arange(len(vocab), device="cuda")
        a.plain_ce = True                           # head_ce has no ignore_index
    else:
        vocab, head, remap = None, w.head, None

    def kl(pred, want):                             # KL(Kolibri || drafter) over the slice, Kolibri's own head on its state
        logp = F.log_softmax((want.to(torch.bfloat16) @ head.T).float(), -1)
        logq = F.log_softmax((pred.to(torch.bfloat16) @ head.T).float(), -1)
        return (logp.exp() * (logp - logq)).sum(-1).mean()
    history, decay_at, decay_len = [], None, 0

    def lr_at(s: int) -> float:                     # warmup, constant, then a linear decay once held-out plateaus
        f = min(1.0, (s + 1) / 100)
        if decay_at is not None:
            f *= max(0.05, 1 - (s - decay_at) / max(1, decay_len))
        return a.lr * f

    def check(s: int) -> bool:
        """A held-out check; True when training should stop (the decay phase has run out)."""

        nonlocal decay_at, decay_len
        dr.eval()
        r = held.score(dr, w.embed, w.head)
        dr.train()
        history.append(r["accepted"])
        print(json.dumps({"step": s, "held_out": r, "decaying": decay_at is not None}), flush=True)
        if decay_at is None and len(history) >= 4 and max(history[-3:]) - history[-4] < a.plateau:
            decay_at, decay_len = s, max(300, int(a.decay * s))
            print(json.dumps({"step": s, "plateau": history[-4:], "decay_steps": decay_len}), flush=True)
        return decay_at is not None and s >= decay_at + decay_len

    class _Sched:
        def step(self):
            pass

        def get_last_lr(self):
            return [opt.param_groups[0]["lr"]]

    sched = _Sched()
    stop = False
    rng = np.random.default_rng(0)
    t0, seen, step, log = time.perf_counter(), 0, 0, {"l1": 0.0, "ce": 0.0, "acc": 0.0, "n": 0}
    for epoch in range(a.epochs):
        for ci in rng.permutation(len(train)):
            name, toks = train[ci]
            seq = torch.from_numpy(np.asarray(toks, dtype=np.int64)).cuda()
            with torch.no_grad():
                x, taps, picks = target_pass(model, seq.tolist())
            seen += len(toks)
            if a.max_tokens and seen > a.max_tokens:
                stop = True
                break
            for s0 in range(0, x.shape[0], a.window):
                batch = chain_batch(x, taps, seq, picks, w.embed, s0, a.window, a.rollout)
                if batch is None:
                    continue
                feats, embs, wants, labels, _ = batch
                if a.noise:
                    feats = feats + ((torch.rand(feats.shape, device=feats.device) * 2 - 1) * a.noise).to(feats.dtype)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = dr.rollout(feats, embs)
                weights = [0.8 ** j for j in range(len(outs))]
                l1 = sum(wt * F.smooth_l1_loss(o.float() / unit, want.float() / unit)
                         for wt, o, want in zip(weights, outs, wants)) / sum(weights)
                if remap is not None:
                    labels = [remap[lab] for lab in labels]
                if a.loss == "l1ce":
                    ce_of = plain_ce if a.plain_ce else head_ce
                    ce = sum(wt * ce_of(o, head, lab) for wt, o, lab in zip(weights, outs, labels)) / sum(weights)
                    loss = a.l1w * l1 + a.ce * ce
                else:
                    ce = sum(wt * kl(o, want) for wt, o, want in zip(weights, outs, wants)) / sum(weights)
                    loss = ce + (a.l1w * l1 if a.loss == "kll1" else 0.0)
                pred, label = outs[0], labels[0]
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dr.parameters(), 0.5)
                opt.step()
                sched.step()
                step += 1
                with torch.no_grad():
                    k = min(256, pred.shape[0])
                    acc = ((pred[-k:].to(torch.bfloat16) @ head.T).argmax(-1) == label[-k:]).float().mean()
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
                    torch.save({"config": cfg.__dict__, "state": dr.state_dict(), "step": step, "tokens": seen,
                                "taps": TAPS}, out / "drafter.pt")
                if held is not None and step % a.eval == 0 and check(step):
                    stop = True
                    break
            if stop:
                break
        if stop:
            break
    torch.save({"config": cfg.__dict__, "state": dr.state_dict(), "step": step, "tokens": seen, "taps": TAPS},
               out / "drafter.pt")
    if held is not None:
        print(json.dumps({"final": held.score(dr.eval(), w.embed, w.head), "step": step}), flush=True)


if __name__ == "__main__":
    main()
