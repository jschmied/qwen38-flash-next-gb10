"""The held-out conversations in TENSORFOLD_KOLIBRI_RECORD's format, from one offline Kolibri-1 pass (prompt rows).

    python record_heldout.py MODEL_DIR PREFIX NAMES.json OUT_DIR [--taps 44,47,49] [--max 12000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import assistant_mask, conversations  # noqa: E402


@torch.no_grad()
def main() -> None:
    from tokenizers import Tokenizer

    from tensorfold.families.kolibri1.cuda.forward import Chain, Model
    from tensorfold.families.kolibri1.cuda.record import Recorder
    from tensorfold.families.kolibri1.cuda.weights import load

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("prefix")
    ap.add_argument("names")
    ap.add_argument("out")
    ap.add_argument("--taps", default="44,47,49")
    ap.add_argument("--max", type=int, default=12000)
    a = ap.parse_args()
    taps = [int(v) for v in a.taps.split(",")]
    names = set(json.loads(Path(a.names).read_text()))
    w = load(a.model_dir)
    model = Model(w, a.max + 64, 1)
    model.record_taps = tuple(taps)
    rec = Recorder(a.out, taps, k=32, floor_gb=0)
    tok = Tokenizer.from_file(str(Path(a.model_dir) / "tokenizer.json"))
    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    header = tok.encode("assistant\n", add_special_tokens=False).ids
    for sid, (name, toks) in enumerate(c for c in conversations(a.prefix) if c[0] in names):
        ids = [int(t) for t in toks[:a.max]]
        kinds = assistant_mask(np.asarray(ids), start, header, end).astype(np.uint8)   # 1: Kolibri wrote it
        for p in range(0, len(ids), 8192):
            model.forward([Chain(0, p, ids[p:p + 8192])], prompt=True)
            final, states = model.last
            rec.add(sid, range(p, p + len(ids[p:p + 8192])), ids[p:p + 8192], states, final, w.head,
                    kinds=kinds[p:p + 8192].tolist())
        rec.finish(sid, source=name)
        print(f"{name}: {len(ids)} rows, {int(kinds.sum())} written by Kolibri", flush=True)


if __name__ == "__main__":
    main()
