"""Kolibri-1 answers first user turns from chat sets (its own replies, for drafter training), appended as JSON lines.

    python gen_chat.py URL OUT.jsonl SOURCE [SOURCE ...] [--workers 2] [--max-tokens 3000] [--hours 12]

Prompts go round-robin over the sources (data.py's shapes); a prompt already in OUT is skipped, so a restart resumes.
Sampling is the server's default (Kolibri's generation config); thinking on. Each line holds the messages with the
assistant's reasoning_content, in the shape data_swe.py renders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data import messages, rows  # noqa: E402


def prompts(sources: list[str], max_chars: int = 12000, max_rows: int = 0):
    """First user turns, round-robin over the sources; each source stops after its first `max_rows` rows (0: all), so
    the reserved test rows (sharegpt-deutsch 1,500+, alpaca-gpt4-de 8,000+, ultrachat 60,000+) are never read."""

    its = [rows(Path(p)) for p in sources]
    read = [0] * len(its)
    while its:
        for i, it in list(enumerate(its)):
            r = next(it, None) if it is not None and (not max_rows or read[i] < max_rows) else None
            if r is None:
                its[i] = None
                continue
            read[i] += 1
            msgs = messages(r)
            first = next((m for m in msgs or [] if m["role"] == "user"), None)
            if first and 0 < len(first["content"]) <= max_chars:
                yield first["content"]
        if all(x is None for x in its):
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--skip", nargs="*", default=[], help="other OUT files whose prompts count as done")
    ap.add_argument("--max-rows", type=int, default=1400, help="rows read per source at most (test pool beyond)")
    a = ap.parse_args()
    out = Path(a.out)
    done = set()
    for f in [out, *map(Path, a.skip)]:
        if f.exists():
            for line in f.read_text().splitlines():
                done.add(json.loads(line)["prompt_sha"])
    feed = prompts(a.sources, max_rows=a.max_rows)
    lock = threading.Lock()
    stop_at = time.time() + a.hours * 3600
    stats = {"done": 0, "tokens": 0, "errors": 0}

    def take():
        with lock:
            for p in feed:
                h = hashlib.sha256(p.encode()).hexdigest()[:16]
                if h not in done:
                    done.add(h)
                    return p, h
        return None

    def work():
        while time.time() < stop_at:
            job = take()
            if job is None:
                return
            prompt, h = job
            body = {"model": "kolibri-1", "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": a.max_tokens}
            try:
                req = urllib.request.Request(a.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
                r = json.load(urllib.request.urlopen(req, timeout=3600))
            except Exception as exc:          # noqa: BLE001  (one prompt fails, the night goes on)
                with lock:
                    stats["errors"] += 1
                print(f"error: {exc}", flush=True)
                time.sleep(5)
                continue
            msg = r["choices"][0]["message"]
            line = {"prompt_sha": h, "finish": r["choices"][0]["finish_reason"], "usage": r.get("usage"),
                    "messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": msg.get("content") or "",
                                  "reasoning_content": msg.get("reasoning_content")}]}
            with lock:
                with open(out, "a") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                stats["done"] += 1
                stats["tokens"] += (r.get("usage") or {}).get("completion_tokens", 0)
                if stats["done"] % 20 == 0:
                    print(json.dumps({"time": time.strftime("%H:%M"), **stats}), flush=True)

    threads = [threading.Thread(target=work) for _ in range(a.workers)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    print(json.dumps({"final": stats}), flush=True)


if __name__ == "__main__":
    main()
