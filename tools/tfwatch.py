"""New commits on ashhart/TensorFold since the last run: every branch's head against the state file.

    python3 tools/tfwatch.py [--state ~/.cache/tfwatch.json]

Prints new and removed branches and, per moved branch, its new commits (sha, author, date, subject, files touched);
prints nothing when nothing moved. The first run only records the heads. Needs `gh` logged in.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

REPO = "ashhart/TensorFold"


def gh(path: str):
    out = subprocess.run(["gh", "api", "--paginate", path], check=True, capture_output=True, text=True).stdout
    # --paginate concatenates JSON arrays as "][", objects one after another
    return json.loads("[" + out.replace("][", ",").strip()[1:-1] + "]") if out.lstrip().startswith("[") else json.loads(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default=str(Path.home() / ".cache/tfwatch.json"))
    a = ap.parse_args()
    state_file = Path(a.state)
    old = json.loads(state_file.read_text()) if state_file.exists() else None
    heads = {b["name"]: b["commit"]["sha"] for b in gh(f"repos/{REPO}/branches?per_page=100")}
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(heads, indent=1))
    if old is None:
        print(f"recorded {len(heads)} branches")
        return
    for name in sorted(set(heads) - set(old)):
        print(f"NEW BRANCH {name} @ {heads[name][:9]}")
    for name in sorted(set(old) - set(heads)):
        print(f"REMOVED BRANCH {name} (was {old[name][:9]})")
    for name in sorted(set(heads) & set(old)):
        if heads[name] == old[name]:
            continue
        cmp = gh(f"repos/{REPO}/compare/{old[name]}...{heads[name]}")
        note = " (force-pushed)" if cmp.get("status") in ("diverged", "behind") else ""
        print(f"== {name}: {old[name][:9]} -> {heads[name][:9]}, {cmp.get('ahead_by', '?')} new{note}")
        for c in cmp.get("commits", [])[-30:]:
            m = c["commit"]
            print(f"  {c['sha'][:9]} {m['author']['date'][:16]} {m['author']['name']}: {m['message'].splitlines()[0]}")
        files = [f["filename"] for f in cmp.get("files", [])]
        if files:
            print(f"  files ({len(files)}): " + ", ".join(files[:25]) + (" ..." if len(files) > 25 else ""))


if __name__ == "__main__":
    main()
