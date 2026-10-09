"""Prompts for specbench that no drafter trained on: chat from ultrachat_200k shard 0 rows from 1,400 (gen_chat read
fewer; the test pool starts at 60,000), German from alpaca-gpt4-de rows 7,900-7,999 (batch 4 stopped at 7,900; the
test pool starts at 8,000). Writes specbench's chat file shape: one {"messages": [first user turn]} a line."""

import argparse
import json
from itertools import islice
from pathlib import Path

from data import messages, rows


def first_users(path: Path, start: int, n: int, max_chars: int = 4000) -> list[dict]:
    out = []
    for r in islice(rows(path), start, None):
        m = messages(r)
        if m and m[0]["role"] == "user" and 0 < len(m[0]["content"]) <= max_chars:
            out.append({"messages": [m[0]]})
        if len(out) == n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chat_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    c, o = Path(a.chat_dir), Path(a.out_dir)
    o.mkdir(parents=True, exist_ok=True)
    chat = first_users(c / "HuggingFaceH4__ultrachat_200k__train_sft-00000-of-00003-a3ecf92756993583.parquet", 1400, a.n)
    de = first_users(c / "mayflowergmbh__alpaca-gpt4_de__alpaca_gpt4_data_de.json", 7900, a.n)
    for name, items in (("chat", chat), ("de", de)):
        (o / f"bench_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items))
        print(name, len(items))


if __name__ == "__main__":
    main()
