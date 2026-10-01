#!/usr/bin/env python3
"""How many distinct concurrent-round shapes does Flash Next's MultiDecoder produce? (T12 graph-key scoping)
Wraps gdn_multi.Tables and attn_multi.Step (no behaviour change), runs `streams` concurrent greedy requests of N tokens
on code + prose prompts, prints counts of distinct keys under a coarse and a fine definition.
argv: <tensorfold src> <model dir> <streams> <tokens>"""
import collections, json, sys, threading
from pathlib import Path

src, model, streams, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, src)
import torch                                                                     # noqa: E402
from transformers import AutoTokenizer                                           # noqa: E402

from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi            # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine           # noqa: E402

LOG = {"gdn": [], "attn": [], "mtp": []}
_t0, _s0 = gdn_multi.Tables.__init__, attn_multi.Step.__init__


def tables(self, w, scratch, segs, pending):
    _t0(self, w, scratch, segs, pending)
    LOG["gdn"].append((len(segs), segs[-1][2], tuple(a1 - a0 for _, a0, a1 in segs),
                       tuple(len(p) for p in pending), self.folds, self.cur, self.plan.slots, self.plan.max_rows))


def step(self, w, segs, mtp):
    _s0(self, w, segs, mtp)
    keys = max(self.ends)
    bucket = max(8192, 1 << (keys - 1).bit_length())
    LOG["mtp" if mtp else "attn"].append((self.n, self.rows, self.most, bucket, tuple(self.ends)))


gdn_multi.Tables.__init__, attn_multi.Step.__init__ = tables, step
tok = AutoTokenizer.from_pretrained(model)
TASKS = ["Write a Python module implementing an LRU cache with TTL expiry, type hints and docstrings.",
         "Explain in about 400 words why lighthouses were automated in the twentieth century.",
         "Write a Python class for a thread-safe bounded priority queue with blocking get and put, plus tests.",
         "Schreibe eine ausführliche Erklärung auf Deutsch, wie eine Wärmepumpe im Winter ein Haus heizt."]
prompts = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False), add_special_tokens=False) for t in TASKS[:streams]]
eng = FlashNextEngine(model, max_len=16384, streams=streams)
for k in LOG:
    LOG[k].clear()


def one(i):
    eng.generate(prompts[i], n, None, lambda t: None, stop_eos=False)


th = [threading.Thread(target=one, args=(i,)) for i in range(streams)]
[t.start() for t in th]
[t.join() for t in th]
g = LOG["gdn"]
coarse = {(nn, R, cur, folds, ) for nn, R, lens, held, folds, cur, slots, mr in g}
fine = {(lens, held, cur) for nn, R, lens, held, folds, cur, slots, mr in g}
attn_coarse = {(nn, R, b) for nn, R, most, b, ends in LOG["attn"]}
mtp_coarse = {(nn, R, b) for nn, R, most, b, ends in LOG["mtp"]}
print(json.dumps({"streams": streams, "rounds": len(g), "gdn_keys_coarse": len(coarse), "gdn_keys_fine": len(fine),
                  "attn_keys_coarse": len(attn_coarse), "mtp_steps": len(LOG["mtp"]), "mtp_keys_coarse": len(mtp_coarse),
                  "rows_hist": collections.Counter(R for _, R, *_ in g).most_common(12),
                  "n_hist": collections.Counter(nn for nn, *_ in g).most_common(),
                  "slots_max": max((s for *_, s, _ in g), default=0), "max_rows_hist": collections.Counter(mr for *_, mr in g).most_common()}))
