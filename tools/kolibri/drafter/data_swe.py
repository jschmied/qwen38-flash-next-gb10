"""SWE-bench agent trajectories (mini-swe-agent .traj.json) rendered with Kolibri-1's chat template, one token
sequence a trajectory: TOKENS.bin (uint32, concatenated) and TOKENS.idx.json (offsets, source, per-sequence stats).

    python data_swe.py MODEL_DIR TRAJ_DIR OUT_PREFIX [--max 131072]

Rendered as the server renders the agent's last request plus its reply, with the agent's bash tool declared.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Execute a bash command",
    "parameters": {"type": "object", "properties": {"command": {"type": "string", "description":
                                                                "The bash command to execute"}},
                   "required": ["command"]}}}


def clean(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue
        n = {"role": role, "content": m.get("content") or ""}
        if role == "assistant":
            if m.get("reasoning_content"):
                n["reasoning_content"] = m["reasoning_content"]
            calls = []
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                args = f.get("arguments")
                try:
                    args = json.loads(args) if isinstance(args, str) else (args or {})
                except json.JSONDecodeError:
                    args = {"command": args}
                calls.append({"id": c.get("id"), "type": "function",
                              "function": {"name": f.get("name", "bash"), "arguments": args}})
            if calls:
                n["tool_calls"] = calls
        if role == "tool" and m.get("tool_call_id"):
            n["tool_call_id"] = m["tool_call_id"]
        out.append(n)
    return out


def main() -> None:
    import jinja2
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("traj_dir")
    ap.add_argument("out")
    ap.add_argument("--max", type=int, default=131072, help="tokens a sequence keeps (its start)")
    a = ap.parse_args()
    d = Path(a.model_dir)
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **{k: x for k, x in kw.items()
                                                                                if k in ("indent",)})
    template = env.from_string(json.loads((d / "tokenizer_config.json").read_text())["chat_template"])
    offsets, meta, total = [0], [], 0
    with open(a.out + ".bin", "wb") as f:
        for path in sorted(Path(a.traj_dir).rglob("*.traj.json")):
            try:
                msgs = clean(json.loads(path.read_text()).get("messages") or [])
                if not any(m["role"] == "assistant" for m in msgs):
                    continue
                text = template.render(messages=msgs, tools=[BASH_TOOL], add_generation_prompt=False)
            except Exception as exc:      # noqa: BLE001  (a trajectory that does not render is reported, skipped)
                print(f"skip {path}: {exc}")
                continue
            ids = tok.encode(text, add_special_tokens=False).ids[:a.max]
            np.asarray(ids, dtype=np.uint32).tofile(f)
            total += len(ids)
            offsets.append(total)
            meta.append({"source": str(path.relative_to(a.traj_dir)), "tokens": len(ids),
                         "think": text.count("<think>")})
    Path(a.out + ".idx.json").write_text(json.dumps({"offsets": offsets, "meta": meta}))
    lens = sorted(m["tokens"] for m in meta)
    print(json.dumps({"sequences": len(meta), "tokens": total, "median": lens[len(lens) // 2] if lens else 0,
                      "max": lens[-1] if lens else 0, "think_blocks": sum(m["think"] for m in meta)}))


if __name__ == "__main__":
    main()
