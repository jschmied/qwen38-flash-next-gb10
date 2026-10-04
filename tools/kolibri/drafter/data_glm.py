"""GLM-5.2's regeneration of Open-PerfectBlend (mgoin/open-perfectblend-glm5.2-regen) as whole Kolibri-1 conversations.

    python data_glm.py MODEL_DIR OUT_PREFIX SHARD.parquet [...] [--limit 50000000] [--max-len 16384]

Each conversation is rendered with Kolibri's own chat template, every assistant turn's reasoning kept
(``preserve_thinking``: GLM writes ``reasoning</think>answer``, which the template splits), and stored whole in
data_swe.py's format (OUT_PREFIX.bin uint32 tokens + OUT_PREFIX.idx.json offsets/meta), so train_mt.py prefills each
conversation as serving would. Shards are read round-robin, one conversation at a time, so a limit keeps the mix.
Kolibri's SFT teachers include GLM-5.2 (tech report p53): the closest large stand-in for its own replies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROLE = {"human": "user", "gpt": "assistant", "system": "system"}


def main() -> None:
    import jinja2
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out")
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--limit", type=int, default=50_000_000)
    ap.add_argument("--max-len", type=int, default=16384)
    a = ap.parse_args()
    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        json.loads((d / "tokenizer_config.json").read_text())["chat_template"])

    def rows(path: str):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
            yield from batch.to_pylist()

    its = [(p, rows(p)) for p in a.shards]
    offsets, meta, total, skipped = [0], [], 0, 0
    with open(a.out + ".bin", "wb") as out:
        while its and total < a.limit:
            for i, (shard, it) in list(enumerate(its)):
                r = next(it, None)
                if r is None:
                    its[i] = None
                    continue
                msgs = [{"role": ROLE.get(m["from"], m["from"]), "content": m["value"]} for m in r["conversations"]]
                if not any(m["role"] == "assistant" for m in msgs) or any(m["role"] not in ROLE.values() for m in msgs):
                    skipped += 1
                    continue
                try:
                    text = template.render(messages=msgs, add_generation_prompt=False, preserve_thinking=True)
                except Exception:          # noqa: BLE001  (a malformed conversation is skipped)
                    skipped += 1
                    continue
                ids = tok.encode(text, add_special_tokens=False).ids[:a.max_len]
                np.asarray(ids, dtype=np.uint32).tofile(out)
                total += len(ids)
                offsets.append(total)
                meta.append({"source": f"glm/{Path(shard).stem}/{r.get('source', '')}/{r.get('id', '')}",
                             "tokens": len(ids)})
            its = [x for x in its if x is not None]
    Path(a.out + ".idx.json").write_text(json.dumps({"offsets": offsets, "meta": meta}))
    print(json.dumps({"tokens": total, "conversations": len(meta), "skipped": skipped}))


if __name__ == "__main__":
    main()
