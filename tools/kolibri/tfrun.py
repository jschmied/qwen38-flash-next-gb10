"""Kolibri-1 on TensorFold's CUDA family: a greedy reply against ref.py's saved one, then decode and prefill timings.

    python tfrun.py MODEL_DIR REF.pt [--new 24] [--decode 128] [--prefill 2048,8192]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("ref")
    ap.add_argument("--decode", type=int, default=128)
    ap.add_argument("--prefill", default="2048,8192")
    ap.add_argument("--context", type=int, default=16384)
    a = ap.parse_args()
    from tokenizers import Tokenizer

    from tensorfold.families.kolibri1.cuda.engine import Kolibri1Engine

    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    ref = torch.load(a.ref)
    eng = Kolibri1Engine(d, context=a.context)
    m = eng.model
    prompt, want = ref["prompt"], ref["reply"]
    # greedy, logits row by row: the first from the prompt, then one decode row per reference token
    logits = m.prefill(prompt)
    rows, got = [logits[0]], []
    for i in range(len(want)):
        t = int(rows[-1].argmax())
        got.append(t)
        if i == len(want) - 1:
            break
        rows.append(m.forward([t], len(prompt) + i, prompt=False)[0])
    print("TF REPLY:", repr(tok.decode(got, skip_special_tokens=False)), flush=True)
    print("REF     :", repr(tok.decode(want, skip_special_tokens=False)), flush=True)
    same = next((i for i, (x, y) in enumerate(zip(got, want)) if x != y), len(want))
    print(f"tokens equal to the fp32 reference: {same} of {len(want)}", flush=True)
    rl = ref["logits"].cuda()
    n = min(len(rows), rl.shape[0], same + 1)
    tf = torch.stack(rows[:n]).float()
    lp, lq = torch.log_softmax(rl[:n].float(), -1), torch.log_softmax(tf, -1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    print("KL(ref||tf) per row:", [round(float(x), 4) for x in kl], flush=True)
    # decode speed: one row a forward
    pos = len(prompt) + len(got)
    t = got[-1]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(a.decode):
        t = int(m.forward([t], pos + i, prompt=False)[0].argmax())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"decode: {a.decode} tokens in {dt:.2f}s = {a.decode / dt:.1f} tok/s (one row a forward, eager)", flush=True)
    for n in [int(x) for x in a.prefill.split(",") if x]:
        ids = torch.randint(1000, 120000, (n,), generator=torch.Generator().manual_seed(n)).tolist()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.prefill(ids)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"prefill {n}: {dt:.2f}s = {n / dt:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
