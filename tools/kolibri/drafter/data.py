"""Conversations rendered with Kolibri-1's own chat template, packed into fixed-length uint32 token sequences.

    python data.py MODEL_DIR OUT.bin --seq 2048 --limit 60000000 SOURCE [SOURCE ...]

A SOURCE is a parquet or json(l) file in one of the shapes: ``messages`` [{role, content}] (UltraChat),
``conversations`` [{from, value}] (ShareGPT), instruction/input/output (Alpaca), instruction/response (OSS-Instruct).
Sources are interleaved one conversation at a time, so a limit keeps the mix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path: Path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
            yield from batch.to_pylist()
    elif path.suffix == ".jsonl":
        with open(path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        yield from (data if isinstance(data, list) else data.get("data", []))


ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}


def messages(r: dict) -> list[dict] | None:
    if "messages" in r:
        return [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
    if "conversations" in r:
        out = [{"role": ROLE.get(m.get("from"), ""), "content": m.get("value", "")} for m in r["conversations"]]
        return out if all(m["role"] for m in out) else None
    if "output" in r and "instruction" in r:
        user = r["instruction"] + (("\n\n" + r["input"]) if r.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": r["output"]}]
    if "response" in r and "instruction" in r:
        return [{"role": "user", "content": r["instruction"]}, {"role": "assistant", "content": r["response"]}]
    return None


def main() -> None:
    import jinja2
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=60_000_000)
    a = ap.parse_args()
    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads((d / "tokenizer_config.json").read_text())["chat_template"])
    its = [rows(Path(p)) for p in a.sources]
    buf: list[int] = []
    seqs, counts, total = [], [0] * len(its), 0
    with open(a.out, "wb") as out:
        while its and total < a.limit:
            for i, it in list(enumerate(its)):
                r = next(it, None)
                if r is None:
                    its[i] = None
                    continue
                msgs = messages(r)
                if not msgs or not any(m["role"] == "assistant" for m in msgs):
                    continue
                try:
                    text = template.render(messages=msgs, add_generation_prompt=False, reasoning_effort="none")
                except Exception:          # noqa: BLE001  (a malformed conversation is skipped)
                    continue
                buf += tok.encode(text, add_special_tokens=False).ids
                counts[i] += 1
                while len(buf) >= a.seq:
                    np.asarray(buf[:a.seq], dtype=np.uint32).tofile(out)
                    buf = buf[a.seq:]
                    total += a.seq
            its = [x for x in its if x is not None]
    print(json.dumps({"tokens": total, "sequences": total // a.seq, "seq": a.seq,
                      "conversations": dict(zip(a.sources, counts))}))


if __name__ == "__main__":
    main()
