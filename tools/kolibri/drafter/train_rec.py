"""Drafter training from serving recordings (TENSORFOLD_KOLIBRI_RECORD): no Kolibri copy, only its embedding and head.

    python train_rec.py MODEL_DIR OUT_DIR --data DIR [DIR ...] --held DIR [--init drafter.pt] [--layers 1] [--rollout 3]

A recording holds, per kept row in position order, its input token, the tapped layer states (the last tap is the
head's state) and Kolibri's top-k next-token log-probs. Chain t reads the tapped states at t and the token at t + 1;
step j is trained against the head's state at t + 1 + j (L1) and Kolibri's top-k distribution there (soft
cross-entropy). New recordings in --data are picked up between passes, so training can run beside serving.
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

HEAD_CHUNK = 512


def head_weights(model_dir: str):
    """Kolibri's embedding [V, D] and head [V, D] (bf16), read alone from the checkpoint."""

    from safetensors import safe_open

    d = Path(model_dir)
    index = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(str(d / index[name]), framework="pt", device="cuda") as f:
            out[name] = f.get_tensor(name).to(torch.bfloat16).contiguous()
    return out["model.embed_tokens.weight"], out["lm_head.weight"]


def recordings(dirs: list[str]) -> list[dict]:
    """Finished recordings (those with a .json), as memory maps."""

    runs = []
    for d in dirs:
        for meta in sorted(Path(d).rglob("*.json")):
            info = json.loads(meta.read_text())
            base, n, k = str(meta)[:-5], info["rows"], info["k"]
            width = len(info["taps"]) * info["dims"]
            runs.append({"name": base, "n": n, "taps": info["taps"],
                         "tok": np.memmap(base + ".tok", dtype=np.int32, mode="r", shape=(n,)),
                         "st": np.memmap(base + ".st", dtype=np.int16, mode="r", shape=(n, width)),
                         "tki": np.memmap(base + ".tki", dtype=np.int32, mode="r", shape=(n, k)),
                         "tkl": np.memmap(base + ".tkl", dtype=np.int16, mode="r", shape=(n, k))})
    return runs


def window(run: dict, s0: int, size: int, steps: int, dims: int):
    """Chains t in [s0, s0 + size) with all targets: tapped states, step tokens, head states, top-k ids and probs."""

    t1 = min(s0 + size, run["n"] - steps - 1)
    if t1 <= s0:
        return None
    hi = t1 + steps + 1
    st = torch.from_numpy(np.array(run["st"][s0:hi])).cuda().view(torch.bfloat16)
    tok = torch.from_numpy(np.array(run["tok"][s0:hi], dtype=np.int64)).cuda()
    tki = torch.from_numpy(np.array(run["tki"][s0:hi], dtype=np.int64)).cuda()
    tkp = torch.from_numpy(np.array(run["tkl"][s0:hi])).cuda().view(torch.float16).float().softmax(-1)
    n = t1 - s0
    head_state = st[:, -dims:]                         # the last tap is the head's own state
    return (st[:n], [tok[1 + j:1 + j + n] for j in range(steps)], [head_state[1 + j:1 + j + n] for j in range(steps)],
            [tki[1 + j:1 + j + n] for j in range(steps)], [tkp[1 + j:1 + j + n] for j in range(steps)])


def soft_ce(pred: torch.Tensor, head: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -sum_k p_k log q(id_k), q the drafter's full-vocabulary softmax (row chunks, checkpointed)."""

    from torch.utils.checkpoint import checkpoint

    def part(p, i, w):
        logq = (p.to(torch.bfloat16) @ head.T).float().log_softmax(-1)
        return -(logq.gather(1, i) * w).sum()

    total = sum(checkpoint(part, pred[r:r + HEAD_CHUNK], ids[r:r + HEAD_CHUNK], probs[r:r + HEAD_CHUNK],
                           use_reentrant=False) for r in range(0, pred.shape[0], HEAD_CHUNK))
    return total / pred.shape[0]


@torch.no_grad()
def held_score(dr, runs, embed, head, steps: int, size: int, dims: int) -> dict:
    """Per-step top-1 against Kolibri's choice, and mean drafts accepted in a row, over every row of ``runs``."""

    n, accepted, hits = 0, 0.0, [0] * steps
    for run in runs:
        for s0 in range(0, run["n"], size):
            b = window(run, s0, size, steps, dims)
            if b is None:
                continue
            feats, toks, _, ids, _ = b
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = dr.rollout(feats, [embed[t] for t in toks])
            ok_run = torch.ones(feats.shape[0], dtype=torch.bool, device=feats.device)
            for j, o in enumerate(outs):
                got = torch.cat([(o[i:i + HEAD_CHUNK].to(torch.bfloat16) @ head.T).argmax(-1)
                                 for i in range(0, o.shape[0], HEAD_CHUNK)])
                ok = got == ids[j][:, 0]
                hits[j] += int(ok.sum())
                ok_run &= ok
                accepted += float(ok_run.sum())
            n += feats.shape[0]
    return {"accepted": round(accepted / max(1, n), 4), "top1_steps": [round(h / max(1, n), 4) for h in hits], "rows": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--ffn", type=int, default=4096)
    ap.add_argument("--rollout", type=int, default=3)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1", type=float, default=1.0, help="weight of the head-state L1 beside the soft cross-entropy")
    ap.add_argument("--eval", type=int, default=250)
    ap.add_argument("--save", type=int, default=500)
    ap.add_argument("--log", type=int, default=50)
    ap.add_argument("--hours", type=float, default=24.0, help="stop after this long (passes repeat with new data)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    embed, head = head_weights(a.model_dir)
    held = recordings([a.held])
    if not held:
        raise SystemExit(f"no held-out recordings under {a.held}")
    taps, dims = held[0]["taps"], embed.shape[1]
    cfg = DraftConfig(hidden=dims, layers=a.layers, ffn=a.ffn, taps=len(taps))
    if a.init:                                       # warm start: the fuse passes the last tap (the head's state)
        ck = torch.load(a.init, map_location="cuda")
        old = load_drafter(ck, a.layers)
        dr = Drafter(DraftConfig(**{**old.cfg.__dict__, "taps": len(taps)}))
        missing, unexpected = dr.load_state_dict(old.state_dict(), strict=False)
        assert not unexpected and all(m.startswith("fuse") or m.startswith("blocks.") for m in missing), missing
        with torch.no_grad():
            dr.fuse.weight.zero_()
            dr.fuse.weight[:, -dims:] = torch.eye(dims)
        cfg = dr.cfg
    else:
        dr = Drafter(cfg)
    dr = dr.cuda()
    probe = torch.from_numpy(np.asarray(held[0]["st"][:2048, -dims:])).cuda().view(torch.bfloat16).float()
    unit = float(probe.pow(2).mean().sqrt())            # the head state's RMS (~43 on Kolibri): L1 in that unit
    opt = torch.optim.AdamW(dr.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    print(f"drafter: {params(dr) / 1e6:.1f}M trainable, taps {taps}, held-out rows {sum(r['n'] for r in held)}",
          flush=True)
    dr.eval()
    print(json.dumps({"step": 0, "held_out": held_score(dr, held, embed, head, a.rollout, a.window, dims)}), flush=True)
    dr.train()
    rng = np.random.default_rng(0)
    t0, step, seen, log = time.perf_counter(), 0, 0, {"l1": 0.0, "ce": 0.0, "n": 0}
    stop_at = t0 + a.hours * 3600
    while time.perf_counter() < stop_at:
        runs = recordings(a.data)
        if not runs:
            print("waiting for recordings", flush=True)
            time.sleep(300)
            continue
        for ri in rng.permutation(len(runs)):
            run = runs[ri]
            for s0 in range(0, run["n"], a.window):
                b = window(run, s0, a.window, a.rollout, dims)
                if b is None:
                    continue
                feats, toks, wants, ids, probs = b
                lr = a.lr * min(1.0, (step + 1) / 100)
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outs = dr.rollout(feats, [embed[t] for t in toks])
                weights = [0.8 ** j for j in range(len(outs))]
                l1 = sum(wt * F.smooth_l1_loss(o.float() / unit, want.float() / unit)
                         for wt, o, want in zip(weights, outs, wants)) / sum(weights)
                ce = sum(wt * soft_ce(o, head, i, p) for wt, o, i, p in zip(weights, outs, ids, probs)) / sum(weights)
                loss = a.l1 * l1 + ce
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dr.parameters(), 0.5)
                opt.step()
                step += 1
                seen += feats.shape[0]
                log["l1"] += float(l1.detach())
                log["ce"] += float(ce.detach())
                log["n"] += 1
                if step % a.log == 0:
                    n = log["n"]
                    print(json.dumps({"step": step, "rows": seen, "runs": len(runs), "l1": round(log["l1"] / n, 4),
                                      "ce": round(log["ce"] / n, 3), "h": round((time.perf_counter() - t0) / 3600, 2)}),
                          flush=True)
                    log = {"l1": 0.0, "ce": 0.0, "n": 0}
                if step % a.eval == 0:
                    dr.eval()
                    print(json.dumps({"step": step, "held_out": held_score(dr, held, embed, head, a.rollout, a.window,
                                                                           dims)}), flush=True)
                    dr.train()
                if step % a.save == 0:
                    torch.save({"config": cfg.__dict__, "state": dr.state_dict(), "step": step, "rows": seen,
                                "taps": taps}, out / "drafter.pt")
                if time.perf_counter() >= stop_at:
                    break
    torch.save({"config": cfg.__dict__, "state": dr.state_dict(), "step": step, "rows": seen, "taps": taps},
               out / "drafter.pt")


if __name__ == "__main__":
    main()
