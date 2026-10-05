"""Chat-shaped text (no reasoning) as whole Kolibri-1 conversations for train_mt.py's prefill path.

    python data_text.py MODEL_DIR OUT_PREFIX SOURCE[:SKIP] [...] [--limit 20000000] [--max-len 16384] [--effort none]

A SOURCE is any shape data.py reads (messages / ShareGPT conversations / Alpaca / OSS-Instruct); ``:SKIP`` drops its
first SKIP rows — gen_chat.py walks the same files from the start, so the prompts Kolibri answers itself (and the
recorded German held-out) stay out of this text. Rendered with Kolibri's template at ``--effort`` (default none: these
answers carry no reasoning, and Kolibri writes an empty reasoning block in that mode), stored in data_swe.py's format.
Sources are interleaved one conversation at a time.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from data import messages, rows


def main() -> None:
    import jinja2
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--limit", type=int, default=20_000_000)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--effort", default="none")
    a = ap.parse_args()
    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads((d / "tokenizer_config.json").read_text())["chat_template"])
    its = []
    for s in a.sources:
        path, _, skip = s.partition(":")
        its.append((path, itertools.islice(rows(Path(path)), int(skip or 0), None)))
    offsets, meta, total, skipped = [0], [], 0, 0
    with open(a.out + ".bin", "wb") as out:
        while its and total < a.limit:
            for i, (path, it) in list(enumerate(its)):
                r = next(it, None)
                if r is None:
                    its[i] = None
                    continue
                msgs = messages(r)
                if not msgs or not any(m["role"] == "assistant" for m in msgs):
                    skipped += 1
                    continue
                try:
                    text = template.render(messages=msgs, add_generation_prompt=False, reasoning_effort=a.effort)
                except Exception:          # noqa: BLE001  (a malformed conversation is skipped)
                    skipped += 1
                    continue
                ids = tok.encode(text, add_special_tokens=False).ids[:a.max_len]
                np.asarray(ids, dtype=np.uint32).tofile(out)
                total += len(ids)
                offsets.append(total)
                meta.append({"source": f"text/{Path(path).stem}", "tokens": len(ids)})
            its = [x for x in its if x is not None]
    Path(a.out + ".idx.json").write_text(json.dumps({"offsets": offsets, "meta": meta}))
    print(json.dumps({"tokens": total, "conversations": len(meta), "skipped": skipped}))


if __name__ == "__main__":
    main()
