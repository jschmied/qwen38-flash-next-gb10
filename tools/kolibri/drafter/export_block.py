"""A release directory from a train_block.py checkpoint, in run 11's release shape: model.safetensors (bf16),
config.json (the drafter's config, taps, serving defaults, provenance) and draft_vocab.json.

    python export_block.py CHECKPOINT.pt VOCAB.json OUT_DIR [--name NAME]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("vocab")
    ap.add_argument("out")
    ap.add_argument("--target", default="Aleph-Alpha/Kolibri-1")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.to(torch.bfloat16).contiguous() for k, v in ck["state"].items()}
    save_file(state, out / "model.safetensors")
    shutil.copy(a.vocab, out / "draft_vocab.json")
    vocab = json.loads(Path(a.vocab).read_text())
    meta = {
        "architecture": "kolibri1-block-drafter",
        "target": a.target,
        "drafter": ck["config"],
        "taps": list(ck["taps"]),
        "draft_vocab": "draft_vocab.json",
        "draft_vocab_size": len(vocab),
        "serving": {"depth": 2, "quant": "fp8", "head": "nvfp4"},   # measured best on GB10 (specbench, 2026-10-09)
        "training_tokens": int(ck.get("tokens", 0)),
        "training_steps": int(ck.get("step", 0)),
    }
    (out / "config.json").write_text(json.dumps(meta, indent=1))
    print(f"{out}: {sum(v.numel() for v in state.values()) / 1e6:.1f}M parameters, vocab {len(vocab)}")


if __name__ == "__main__":
    main()
