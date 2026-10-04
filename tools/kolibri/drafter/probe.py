"""Which Kolibri-1 layers know the next tokens: a ridge probe from each layer's output to the target's own state k ahead.

    python probe.py MODEL_DIR PREFIX NAMES.json [--max 12000] [--sets "12,25,49;9,24,49"]

For each layer l and k = 1..3: fit, on all but the last held-out conversation, a linear map from layer l's output at
position t to Kolibri's final state at t + k - 1 (the state that chooses token t + k); score top-1 through Kolibri's
own head on the last conversation's assistant tokens. Then the same for sets of layers, their outputs concatenated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import HEAD_CHUNK, assistant_mask, conversations  # noqa: E402


def ridge(x: torch.Tensor, y: torch.Tensor, lam: float = 1e-3):
    """(W, mx, my): y ~ (x - mx) W + my, fp64 normal equations with lam x mean diagonal."""

    x, y = x.double(), y.double()
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    g = xc.T @ xc
    g += torch.eye(g.shape[0], device=g.device, dtype=g.dtype) * lam * g.diagonal().mean()
    return torch.linalg.solve(g, xc.T @ yc), mx, my


@torch.no_grad()
def main() -> None:
    from tokenizers import Tokenizer

    from tensorfold.cuda import moe as shared
    from tensorfold.families.kolibri1.cuda.forward import Chain, Model
    from tensorfold.families.kolibri1.cuda.weights import load

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("prefix")
    ap.add_argument("names")
    ap.add_argument("--max", type=int, default=12000)
    ap.add_argument("--sets", default="12,25,49;9,24,49;4,24,49;24,39,49;14,29,44;34,44,49;19,34,49;39,44,49;4,9,49")
    a = ap.parse_args()
    names = set(json.loads(Path(a.names).read_text()))
    convs = [c for c in conversations(a.prefix) if c[0] in names]
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    header = tok.encode("assistant\n", add_special_tokens=False).ids
    w = load(a.model_dir)
    layers = list(range(w.config.layers))
    model = Model(w, a.max + 64, 1)
    data = []                                       # per conversation: states [L][T, D] bf16, final x, picks, mask
    for name, toks in convs:
        ids = np.asarray(toks[:a.max], dtype=np.int64)
        st, xs = [[] for _ in layers], []
        for p in range(0, len(ids), 8192):
            x, s = model.forward([Chain(0, p, ids[p:p + 8192].tolist())], prompt=True, features=True, taps=layers)
            xs.append(x)
            for l, v in enumerate(s):
                st[l].append(v)
        x = torch.cat(xs)
        picks = torch.cat([shared.router(x[i:i + HEAD_CHUNK].contiguous(), w.head).argmax(-1)
                           for i in range(0, x.shape[0], HEAD_CHUNK)])
        data.append(([torch.cat(v) for v in st], x, picks, torch.from_numpy(assistant_mask(ids, start, header, end))
                     .cuda()))
        print(f"{name}: {len(ids)} tokens", flush=True)
    del model
    torch.cuda.empty_cache()
    train, test = data[:-1], data[-1]

    def pairs(conv, feats, k):
        """Inputs at t, the final state at t + k - 1, its choice (token t + k), assistant mask at t + k."""

        _, x, picks, mask = conv
        n = x.shape[0] - k
        return feats[:n], x[k - 1:k - 1 + n], picks[k - 1:k - 1 + n], mask[k:k + n]

    def score(feat_of) -> list[float]:
        out = []
        for k in (1, 2, 3):
            xs, ys = zip(*[pairs(c, feat_of(c), k)[:2] for c in train])
            wgt, mx, my = ridge(torch.cat(xs).float(), torch.cat(ys).float())
            fx, _, lab, m = pairs(test, feat_of(test), k)
            pred = ((fx.double() - mx) @ wgt + my).to(torch.bfloat16)
            got = torch.cat([(pred[i:i + HEAD_CHUNK] @ w.head.T).argmax(-1) for i in range(0, pred.shape[0], HEAD_CHUNK)])
            out.append(round(float(((got == lab) & m).sum() / m.sum()), 4))
        return out

    results = {"layers": {}, "sets": {}}
    for l in layers:
        results["layers"][l] = score(lambda c, l=l: c[0][l])
        print(json.dumps({"layer": l, "top1_k1_k2_k3": results["layers"][l]}), flush=True)
    for spec in a.sets.split(";"):
        ls = [int(v) for v in spec.split(",")]
        results["sets"][spec] = score(lambda c, ls=ls: torch.cat([c[0][l] for l in ls], -1))
        print(json.dumps({"set": spec, "top1_k1_k2_k3": results["sets"][spec]}), flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
