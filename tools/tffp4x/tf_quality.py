#!/usr/bin/env python3
"""Teacher-forced quality of Flash Next on TensorFold under --precision full vs checkpoint: 8 sequences of 4,096
tokens (wikitext-2 test x 4, CPython source x 4), fed 8 tokens a step through the decode path, every position scored.
The full arm writes each position's true-token log-prob, argmax and its top-32 (ids, log-probs); the checkpoint arm
reads them and writes its own true-token log-prob, argmax and log-probs at the reference's top-32 (for a KL estimate
with the tail as one bucket). argv: <tensorfold src> <model dir> <mode> <wikitext text> <out.pt> [<reference.pt>]"""
import sys, sysconfig, time
from pathlib import Path

src, model, mode, wt, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), sys.argv[5]
ref_path = sys.argv[6] if len(sys.argv) > 6 else None
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.cuda import precision                                            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit                   # noqa: E402

N, STEP, TOPK = 4096, 8, 32
precision.set_mode(mode, asked=True)
tok = AutoTokenizer.from_pretrained(model)
text = tok.encode(wt.read_text())
root = Path(sysconfig.get_paths()["stdlib"])
files = sorted(p for p in root.rglob("*.py") if not any(x in p.parts for x in ("test", "tests", "idlelib", "__pycache__")))
code = tok.encode("".join(f"# {p.relative_to(root)}\n{p.read_text(errors='ignore')}\n" for p in files[600:1400]))
seqs = [text[i * len(text) // 4:][:N + 1] for i in range(4)] + [code[i * len(code) // 4:][:N + 1] for i in range(4)]
eng = FlashNextEngine(model, max_len=N + 64)
e = eng.e
ref = torch.load(ref_path) if ref_path else None
res = []
t0 = time.perf_counter()
for si, seq in enumerate(seqs):
    e.reset()
    true_lp, top1, top_ids, top_lp, at_ref = [], [], [], [], []
    for i in range(0, N, STEP):
        toks = seq[i:i + STEP]
        logits = e.forward(toks).float()
        commit(e.w, e.st, e.buf, len(toks), len(toks))
        lp = torch.log_softmax(logits, dim=-1)
        nxt = torch.tensor(seq[i + 1:i + 1 + len(toks)], device=lp.device)
        true_lp.append(lp.gather(1, nxt[:, None])[:, 0].cpu())
        top1.append(lp.argmax(-1).int().cpu())
        if ref is None:
            v, ix = lp.topk(TOPK, dim=-1)
            top_ids.append(ix.int().cpu())
            top_lp.append(v.cpu())
        else:
            ids = ref["seqs"][si]["top_ids"][i:i + len(toks)].to(lp.device).long()
            at_ref.append(lp.gather(1, ids).cpu())
    row = {"true_lp": torch.cat(true_lp), "top1": torch.cat(top1)}
    if ref is None:
        row.update(top_ids=torch.cat(top_ids), top_lp=torch.cat(top_lp))
    else:
        row["at_ref"] = torch.cat(at_ref)
    res.append(row)
    print(f"seq {si} done {time.perf_counter() - t0:.0f}s nll {float(-row['true_lp'].mean()):.4f}", flush=True)
torch.save({"mode": mode, "seqs": res}, out)
if ref is not None:
    for name, sl in (("wikitext", slice(0, 4)), ("code", slice(4, 8)), ("all", slice(0, 8))):
        kl, agree, nll_r, nll_c, n = 0.0, 0, 0.0, 0.0, 0
        for r, c in zip(ref["seqs"][sl], res[sl]):
            pr = r["top_lp"].exp()
            tail_r = (1 - pr.sum(-1)).clamp_min(1e-12)
            tail_c = (1 - c["at_ref"].exp().sum(-1)).clamp_min(1e-12)
            kl += float((pr * (r["top_lp"] - c["at_ref"])).sum() + (tail_r * (tail_r.log() - tail_c.log())).sum())
            agree += int((r["top1"] == c["top1"]).sum())
            nll_r += float(-r["true_lp"].sum())
            nll_c += float(-c["true_lp"].sum())
            n += r["top1"].numel()
        print(f"RESULT {name}: KL(full||checkpoint) mean {kl / n:.5f}, top-1 agreement {100 * agree / n:.2f} %, "
              f"perplexity full {torch.tensor(nll_r / n).exp():.4f} checkpoint {torch.tensor(nll_c / n).exp():.4f} "
              f"({100 * float(torch.tensor((nll_c - nll_r) / n).exp() - 1):+.2f} %), "
              f"positions {n}", flush=True)
