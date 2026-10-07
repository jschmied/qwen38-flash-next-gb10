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
from model_block import BlockConfig, BlockDrafter, chain_params, from_chain, load, predecessor  # noqa: E402
from model_mt import params  # noqa: E402


def save(dr: BlockDrafter, step: int, seen: int, out: Path) -> None:
    """bf16 weights (half the disk; every save is kept by keepckpt.sh), written whole before the rename."""

    state = {k: v.to(torch.bfloat16) for k, v in dr.state_dict().items()}
    torch.save({"config": dr.cfg.__dict__, "state": state, "step": step, "tokens": seen, "taps": train_mt.TAPS,
                "kind": "block"}, out / "drafter.pt.tmp")
    (out / "drafter.pt.tmp").rename(out / "drafter.pt")


def rec_windows(a, dims: int) -> list:
    """(run, first row) for every window of the training recordings: saved before --rec-before, not the live held-out
    (train_rec.py's every-10th), each window repeated by its passes (SWE agent sessions fewer: they repeat repos)."""

    import zlib

    from tokenizers import Tokenizer
    from train_rec import recordings

    cut = time.mktime(time.strptime(a.rec_before, "%Y-%m-%d %H:%M"))
    tk = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    items, rows = [], {"swe": 0, "chat": 0}
    whole = {str(Path(d).resolve()) for d in a.rec_all}
    for run in recordings(list(a.rec) + list(a.rec_all)):
        in_whole = any(str(Path(run["name"]).resolve()).startswith(w + "/") for w in whole)
        if (not in_whole and Path(run["name"] + ".json").stat().st_mtime > cut) \
                or zlib.crc32(Path(run["name"]).name.encode()) % 10 == 0:
            continue
        if run["taps"] != train_mt.TAPS or run["st"].shape[1] != len(train_mt.TAPS) * dims:
            continue
        text = tk.decode([int(t) for t in run["tok"][:2000]])
        kind = "swe" if "returncode" in text or "interact with a computer shell" in text else "chat"
        passes = a.rec_passes_swe if kind == "swe" else a.rec_passes
        for r0 in range(0, run["n"], a.window):
            items += [(run, r0)] * passes
        rows[kind] += run["n"] * passes
    print(f"recordings: {len(items)} windows, rows with passes {rows}", flush=True)
    return items


def rec_batch(run: dict, r0: int, window: int, block: int, norm: torch.Tensor, dims: int):
    """A recording window as train() takes it: tapped states, the head's normed states (the last tap is layer 49, the
    head's input: final norm and head give Kolibri's top-1 on 98.9 % of rows), tokens, anchors whose t+2 Kolibri wrote."""

    r1 = min(r0 + window, run["n"])
    if r1 - r0 < block + 3:
        return None
    feats = torch.from_numpy(np.array(run["st"][r0:r1])).cuda().view(torch.bfloat16)
    h = feats[:, -dims:].float()
    xs = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * norm.float()).to(torch.bfloat16)
    seq = torch.from_numpy(np.array(run["tok"][r0:r1], dtype=np.int64)).cuda()
    kind = np.asarray(run["kind"][r0:r1], dtype=np.bool_)
    cand = np.arange(0, r1 - r0 - 1 - block)
    cand = cand[kind[cand + 2]]
    return (feats, xs, seq, cand) if len(cand) else None


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
    ap.add_argument("--from-chain", default="", help="a chain drafter (train_mt/train_rec) as position 0 (chain_ctx)")
    ap.add_argument("--pred-rank", type=int, default=0, help="add a predecessor head of this rank (identity at start)")
    ap.add_argument("--row0-chain", action="store_true",
                    help="position 0 IS the chain drafter: its path (fuse, fc, layer 0, out_norm) reloaded from "
                         "--from-chain and frozen; layers 1.. and the mask train for positions 1..")
    ap.add_argument("--decay", default="0.8", help="position weights decay^j; 'a:b:N' eases from a to b over N tokens")
    ap.add_argument("--assistant-only", action="store_true", help="anchors whose first draft (t+2) is the assistant's")
    ap.add_argument("--onpolicy", action="store_true",
                    help="weight position j by Kolibri's probability of the data's tokens t+2..t+1+j (its own path)")
    ap.add_argument("--skip-tokens", type=int, default=0, help="skip the stream's first N tokens (a resumed run)")
    ap.add_argument("--rec", nargs="*", default=[], help="recording dirs (train_rec.py's format) mixed into the stream")
    ap.add_argument("--rec-before", default="2026-10-05 09:55", help="only recordings saved before this (after: scoring)")
    ap.add_argument("--rec-all", nargs="*", default=[], help="recording dirs used whole (none of them is scoring data)")
    ap.add_argument("--rec-share", type=float, default=0.3, help="share of steps from recordings while they last")
    ap.add_argument("--rec-passes", type=int, default=2, help="uses of each chat recording window")
    ap.add_argument("--rec-passes-swe", type=int, default=1, help="uses of each SWE recording window")
    ap.add_argument("--rec-refresh", action="store_true", help="on --resume: a fresh recording pass, not the saved rest")
    ap.add_argument("--loss", default="kl", choices=["kl", "kl-auf", "ce-acc"],
                    help="kl: KL x decay^j (x path); kl-auf: KL while the drafter's own drafts still match Kolibri's "
                         "(accept-until-fail); ce-acc: CE on Kolibri's argmax, position j weighted by d E[accepted] / d a_j")
    ap.add_argument("--accum", type=int, default=1, help="windows per optimizer step")
    ap.add_argument("--resume", default="", help="resume.pt: weights, optimizer, step, data position, rng")
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
    slot = torch.full((w.head.shape[0],), -1, dtype=torch.long, device="cuda")   # full id -> draft-vocab column
    slot[vocab] = torch.arange(len(vocab), device="cuda")
    if a.assistant_only:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
        marks = (tk.token_to_id("<|im_start|>"), tk.encode("assistant\n", add_special_tokens=False).ids,
                 tk.token_to_id("<|im_end|>"))
    cfg = BlockConfig(hidden=w.config.hidden, layers=a.layers, ffn=a.ffn, taps=len(train_mt.TAPS), block=a.block)
    if a.init:
        dr = load(torch.load(a.init, map_location="cuda"), pred_rank=a.pred_rank)
        if a.row0_chain and a.from_chain:                   # put the exact chain back over the loaded weights
            dr.cfg.row0_chain = True
            dr = from_chain(torch.load(a.from_chain, map_location="cpu"), dr.cfg, onto=dr.cpu())
    elif a.from_chain:
        cfg.row0_chain = a.row0_chain
        dr = from_chain(torch.load(a.from_chain, map_location="cpu"), cfg)
    else:
        dr = BlockDrafter(cfg)
    dr = dr.cuda()
    cfg = dr.cfg
    if not a.init and not a.from_chain:
        with torch.no_grad():                             # the target's states carry its final norm's scale (~43)
            dr.out_norm.w.copy_(w.norm.float())
    if cfg.row0_chain:
        for p in chain_params(dr):
            p.requires_grad_(False)
    trainable = [p for p in dr.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    res = torch.load(a.resume, map_location="cuda", weights_only=False) if a.resume else None
    if res:
        dr.load_state_dict(res["state"])
        opt.load_state_dict(res["opt"])
        a.skip_tokens = res["seen"]
    print(f"block drafter: {params(dr) / 1e6:.1f}M trainable, block {cfg.block}, {cfg.layers} layers, taps "
          f"{train_mt.TAPS}; {len(convs)} conversations ({sum(len(t) for _, t in convs) / 1e6:.2f}M tokens)", flush=True)
    d0, d1, span = (float(v) for v in a.decay.split(":")) if ":" in a.decay else (float(a.decay),) * 2 + (1.0,)
    jj = torch.arange(cfg.block, device="cuda", dtype=torch.float32)
    rng, rrng = np.random.default_rng(0), np.random.default_rng(1)
    acc_ema = torch.full((cfg.block,), 0.5, device="cuda")       # each position's match rate given the earlier ones
    t0, step, seen, rec_rows = time.perf_counter(), 0, 0, 0
    log = {"kl": 0.0, "top1": 0.0, "n": 0, "rec": 0}
    skipped, total = 0, sum(len(t) for _, t in convs)
    items = rec_windows(a, w.config.hidden) if a.rec or a.rec_all else []
    rrng.shuffle(items)
    perm = rng.permutation(len(convs))
    if res:                                                       # the same stream, the same draws from here on
        step = res["step"]
        rng.bit_generator.state, rrng.bit_generator.state = res["rng"], res["rrng"]
        if not a.rec_refresh:
            by_name = {run["name"]: run for run, _ in items}
            items = [(by_name[n], r0) for n, r0 in res["items"]]
        acc_ema = res["acc_ema"].cuda()
        print(f"resumed at step {step}, {res['seen']} tokens, {len(items)} recording windows left", flush=True)

    def checkpoint() -> None:
        save(dr, step, seen, out)
        state = {"state": dr.state_dict(), "opt": opt.state_dict(), "step": step, "seen": seen,
                 "rng": rng.bit_generator.state, "rrng": rrng.bit_generator.state, "acc_ema": acc_ema.cpu(),
                 "items": [(run["name"], r0) for run, r0 in items], "credit": credit}
        torch.save(state, out / "resume.pt.tmp")
        (out / "resume.pt.tmp").rename(out / "resume.pt")

    def train(feats, xs, seq, cand, source: str) -> None:
        """One step on a window: feats [T, taps*D] tapped states, xs [T, D] the head's (normed) states, seq [T] tokens,
        cand the anchor rows (window-relative) to draw from."""
        nonlocal step, log, acc_ema
        g = rng if source == "glm" else rrng
        pick = np.sort(g.choice(cand, size=min(a.anchors, len(cand)), replace=False))
        anchors = torch.from_numpy(pick).cuda()
        j = torch.arange(cfg.block, device="cuda")
        tgt_rows = anchors[:, None] + 1 + j[None]                  # Kolibri's state whose head guesses draft j
        with torch.no_grad():
            logp = F.log_softmax((xs[tgt_rows].reshape(-1, xs.shape[1]).to(torch.bfloat16) @ head.T).float(), -1)
        lr = a.lr
        if a.lr_end:
            frac = min(1.0, seen / (a.lr_span or total))
            lr = a.lr_end + (a.lr - a.lr_end) * 0.5 * (1 + math.cos(math.pi * frac))
        for grp in opt.param_groups:
            grp["lr"] = lr * min(1.0, (step + 1) / a.warmup)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            nxt = w.embed[torch.cat([seq[1:], seq[-1:]])] if cfg.chain_ctx else None
            states = dr(feats, w.embed[seq[anchors + 1]], anchors, nxt=nxt)
        if cfg.pred_rank:                                          # each position reads Kolibri's token before it
            with torch.no_grad():
                prev = w.embed[vocab[logp.view(len(pick), cfg.block, -1)[:, :-1].argmax(-1)]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fixed = predecessor(dr, states[:, 1:].reshape(-1, states.shape[-1]), prev.reshape(-1, prev.shape[-1]))
            states = torch.cat([states[:, :1], fixed.view(len(pick), cfg.block - 1, -1)], 1)
        logq = F.log_softmax((states.reshape(-1, states.shape[-1]).to(torch.bfloat16) @ head.T).float(), -1)
        kl = (logp.exp() * (logp - logq)).sum(-1).view(len(pick), cfg.block)
        with torch.no_grad():                                      # P(Kolibri's own path = the data's t+2..t+1+j)
            nxt = slot[seq[tgt_rows[:, :-1] + 1]]
            lp = logp.view(len(pick), cfg.block, -1)[:, :-1].gather(-1, nxt.clamp(min=0)[..., None])[..., 0]
            lp = torch.where(nxt >= 0, lp, torch.full_like(lp, -1e9))
            path = torch.cat([torch.ones_like(lp[:, :1]), lp.cumsum(-1).exp()], 1)
        weights = (d0 + (d1 - d0) * min(1.0, seen / span)) ** jj
        valid = path if a.onpolicy else torch.ones_like(path)
        with torch.no_grad():
            want = logp.view(len(pick), cfg.block, -1).argmax(-1)
            hit = (logq.view(len(pick), cfg.block, -1).argmax(-1) == want).float()          # [A, block]
            alive = torch.cat([torch.ones_like(hit[:, :1]), hit[:, :-1].cumprod(-1)], 1)  # drafts before j all matched
            cond = (hit * alive).sum(0) / alive.sum(0).clamp_min(1)
            acc_ema = 0.98 * acc_ema + 0.02 * cond
        if a.loss == "kl":
            m = weights * valid
            per = kl
        elif a.loss == "kl-auf":
            m = alive * valid
            per = kl
        else:
            r = acc_ema.clamp(1e-3, 1.0)
            pre = torch.cat([torch.ones(1, device="cuda"), r[:-1].cumprod(0)])            # prod_{i<j} a_i
            tail = torch.stack([1 + r.new_zeros(()) + sum((r[j + 1:k + 1].prod() for k in range(j + 1, cfg.block)),
                                                          r.new_zeros(())) for j in range(cfg.block)])
            m = (pre * tail)[None] * valid                                                 # d E[accepted] / d a_j
            per = -logq.view(len(pick), cfg.block, -1).gather(-1, want[..., None])[..., 0]
        if cfg.row0_chain:
            m = torch.cat([torch.zeros_like(m[..., :1]), m[..., 1:]], -1)   # position 0 is frozen: no weight
        loss = (per * m).sum() / m.sum().clamp_min(1e-6)
        (loss / a.accum).backward()
        step += 1
        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 0.5)
            opt.step()
            opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            log["top1"] += float((logq.view(len(pick), cfg.block, -1)[:, 0].argmax(-1)
                                  == logp.view(len(pick), cfg.block, -1)[:, 0].argmax(-1)).float().mean())
        log["kl"] += float(loss.detach())
        log["path"] = log.get("path", 0) + path.mean(0)
        log["n"] += 1
        log["rec"] += source == "rec"
        if step % a.log == 0:
            k, dt = log["n"], time.perf_counter() - t0
            print(json.dumps({"step": step, "tokens": seen, "tok_s": round((seen - skipped) / dt),
                              "kl": round(log["kl"] / k, 4), "top1": round(log["top1"] / k, 3),
                              "path": [round(float(v), 3) for v in log["path"] / k], "rec": log["rec"],
                              "acc": [round(float(v), 3) for v in acc_ema],
                              "rec_rows": rec_rows, "rec_left": len(items), "decay": round(float(weights[1]), 3),
                              "lr": f"{opt.param_groups[0]['lr']:.2e}", "h": round(dt / 3600, 2)}), flush=True)
            log = {"kl": 0.0, "top1": 0.0, "n": 0, "rec": 0}
        if step % a.save == 0:
            checkpoint()

    credit = res["credit"] if res and not a.rec_refresh else 0.0
    for ci in perm:
        name, toks = convs[ci]
        if skipped < a.skip_tokens:                       # same seed, same order: the resumed run's first tokens
            skipped += len(toks)
            seen += len(toks)
            continue
        if a.max_tokens and seen > a.max_tokens:
            break
        seq = torch.from_numpy(np.asarray(toks, dtype=np.int64)).cuda()
        amask = train_mt.assistant_mask(np.asarray(toks), marks[0], marks[1], marks[2]) if a.assistant_only else None
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
            if amask is not None:
                cand = cand[amask[cand + 2]]
                if not len(cand):
                    continue
            train(taps[s0:s1], x[s0:s1], seq[s0:s1], cand - s0, "glm")
            credit += a.rec_share / (1 - a.rec_share) if items else 0.0
            while credit >= 1 and items:                  # Kolibri's own replies, interleaved at --rec-share of steps
                credit -= 1
                run, r0 = items.pop()
                b = rec_batch(run, r0, a.window, cfg.block, w.norm, w.config.hidden)
                if b is not None:
                    rec_rows += b[0].shape[0]
                    train(*b, "rec")
    checkpoint()
    print(json.dumps({"done": step, "tokens": seen}), flush=True)


if __name__ == "__main__":
    main()
