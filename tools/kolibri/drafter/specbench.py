"""End-to-end decode speed of Kolibri-1 on TensorFold with copy drafting alone vs plus the learned drafter.

    python specbench.py MODEL_DIR DRAFTER.pt --vocab V.json --swe TRAJ_DIR --chat CHAT.jsonl --de CHAT_DE.jsonl
        [--depths 1,2,3] [--tokens 256] [--n 8] [--concurrent 4]

Prompts nobody trained on (the caller picks them): SWE turns from mini-swe-agent trajectories (two assistant turns a
trajectory, rendered with Kolibri's template up to that turn), the last ``n`` chat and German prompts. Each arm decodes
the same prompts with the same sampling (greedy, and the serving sampling: SWE 0.6/0.95/20, chat 1.0/0.97/128);
replies must be identical across arms (drafts are verified). Printed per arm and domain: decode tokens/s (prefill
excluded), learned drafts proposed and kept, mismatches. Then ``concurrent`` chat prompts at once: copy vs depth 2.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def prompts(a, template, tok):
    out = {"swe": [], "chat": [], "de": []}
    for f in sorted(Path(a.swe).glob("*.traj.json")):
        msgs = [{"role": m["role"], "content": m["content"]} for m in json.loads(f.read_text())["messages"]]
        turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        for i in (turns[len(turns) // 3], turns[2 * len(turns) // 3]):
            ids = tok.encode(template.render(messages=msgs[:i], add_generation_prompt=True),
                             add_special_tokens=False).ids
            if len(ids) < a.max_prompt or a.test_vocab:
                out["swe"].append(ids)
    for key, path in (("chat", a.chat), ("de", a.de)):
        lines = Path(path).read_text().splitlines()[-a.n:]
        for line in lines:
            user = json.loads(line)["messages"][0]
            out[key].append(tok.encode(template.render(messages=[user], add_generation_prompt=True),
                                       add_special_tokens=False).ids)
    return out


def run(model, drafter, depth, batch, count, sampling):
    """Decode ``batch`` (prompts decoded together); returns replies, decode seconds, tokens, learned proposed/kept."""

    from tensorfold.cuda.streams import Stream
    from tensorfold.families.kolibri1.cuda.decoder import Decoder

    model.record_taps = ()
    if drafter is not None:
        drafter.depth, drafter.cache = depth, {}
    d = Decoder(model, (127906, 127901), drafter=drafter)
    streams = [Stream(list(p), count, sampling, draft=True) for p in batch]
    for s in streams:
        d.admit(s)
    while any(s not in d.streams.values() and not s.done for s in streams):   # prefill first: decode timed alone
        d.round()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while not all(s.done for s in streams):
        d.finish(d.round())
    torch.cuda.synchronize()
    return [list(s.out) for s in streams], time.perf_counter() - t0, sum(len(s.out) for s in streams), \
        d.learned_rows, d.learned_kept


@torch.no_grad()
def main() -> None:
    import jinja2
    from tokenizers import Tokenizer

    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.kolibri1.cuda.drafter import Drafter
    from tensorfold.families.kolibri1.cuda.forward import Model
    from tensorfold.families.kolibri1.cuda.weights import load

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("drafter")
    ap.add_argument("--vocab", default="")
    ap.add_argument("--swe", required=True)
    ap.add_argument("--chat", required=True)
    ap.add_argument("--de", required=True)
    ap.add_argument("--depths", default="1,2,3")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--max-prompt", type=int, default=60000)
    ap.add_argument("--test-vocab", type=int, default=0, help="dry run on a tiny checkpoint: ids folded into its vocab")
    a = ap.parse_args()
    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads((d / "tokenizer_config.json").read_text())["chat_template"])
    P = prompts(a, template, tok)
    if a.test_vocab:
        P = {k: [[t % a.test_vocab for t in p[-300:]] for p in v[:2]] for k, v in P.items()}
        a.max_prompt = 400
    print(json.dumps({k: [len(p) for p in v] for k, v in P.items()}), flush=True)
    w = load(d)
    model = Model(w, a.max_prompt + a.tokens + 64, max(1, a.concurrent))
    dr = Drafter(a.drafter, w.embed, w.head, vocab=a.vocab or None)
    arms = [("copy", None, 0)] + [(f"learned-d{k}", dr, int(k)) for k in a.depths.split(",")]
    served = {"swe": lambda i: Sampling(seed=i, temperature=0.6, top_k=20, top_p=0.95),
              "chat": lambda i: Sampling(seed=i, temperature=1.0, top_k=128, top_p=0.97),
              "de": lambda i: Sampling(seed=i, temperature=1.0, top_k=128, top_p=0.97)}
    for mode in ("greedy", "served"):
        for dom, ps in P.items():
            if not ps:
                continue
            ref = None
            for name, drafter, depth in arms:
                replies, secs, toks, prop, kept = [], 0.0, 0, 0, 0
                for i, p in enumerate(ps):
                    sm = None if mode == "greedy" else served[dom](i)
                    r, s, t, pr, kp = run(model, drafter, depth, [p], a.tokens, sm)
                    replies += r
                    secs, toks, prop, kept = secs + s, toks + t, prop + pr, kept + kp
                ref = ref or replies
                bad = sum(x != y for x, y in zip(replies, ref))
                print(json.dumps({"mode": mode, "domain": dom, "arm": name, "prompts": len(ps), "tok_s": round(toks / max(secs, 1e-9), 2),
                                  "learned_proposed": prop, "learned_kept": kept, "mismatches": bad}), flush=True)
    chat = (P["chat"] + P["de"])[: a.concurrent]
    for name, drafter, depth in (("copy", None, 0), ("learned-d2", dr, 2)):
        _, secs, toks, prop, kept = run(model, drafter, depth, chat, a.tokens, None)
        print(json.dumps({"mode": "greedy", "domain": f"chat+de x{len(chat)} together", "arm": name,
                          "tok_s": round(toks / secs, 2), "learned_proposed": prop, "learned_kept": kept}), flush=True)


if __name__ == "__main__":
    main()
