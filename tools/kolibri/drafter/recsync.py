"""Mirror ~/kolibri-drafter (recordings, drafters, data) to PBS; when local disk runs short, free the oldest recordings
that are verified on PBS by sha256 (never the held-out ones: train_rec.py's every-Nth live held-out, rec/heldout).

    python recsync.py [--every 900] [--low-gb 60] [--free-gb 40] [--live-held 10]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import time
import zlib
from pathlib import Path

ROOT = Path("/home/jschmied/kolibri-drafter")
DEST = "/mnt/bulk/gb10/kolibri-drafter"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-i", "/home/jschmied/.ssh/id_ed25519",
       "-o", "UserKnownHostsFile=/home/jschmied/.ssh/known_hosts", "root@10.0.0.70"]
RSH = "ssh -o BatchMode=yes -i /home/jschmied/.ssh/id_ed25519 -o UserKnownHostsFile=/home/jschmied/.ssh/known_hosts"
PARTS = (".tok", ".pos", ".kind", ".st", ".tki", ".tkl", ".json")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def sync() -> bool:
    cmd = ["rsync", "-a", "--partial", "--exclude", "prep-venv/", "-e", RSH, f"{ROOT}/", f"root@10.0.0.70:{DEST}/"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in (0, 24):                     # 24: files vanished during transfer (a run being written)
        log(f"rsync rc={r.returncode}: {r.stderr.strip()[-300:]}")
    return r.returncode in (0, 24)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def remote_sha(paths: list[str]) -> dict[str, str]:
    if not paths:
        return {}
    r = subprocess.run(SSH + ["cd " + DEST + " && sha256sum " + " ".join(f"'{p}'" for p in paths)],
                       capture_output=True, text=True)
    out = {}
    for line in r.stdout.splitlines():
        h, _, p = line.partition("  ")
        out[p] = h
    return out


def prune(target_gb: float, live_held: int, keep_after: float) -> None:
    runs = sorted((m for m in (ROOT / "rec/live").rglob("*.json")), key=lambda m: m.stat().st_mtime)
    runs = [m for m in runs if m.stat().st_mtime <= keep_after]   # the fresh scoring set stays local
    freed = 0.0
    for meta in runs:
        if freed / 1e9 >= target_gb:
            break
        base = str(meta)[:-5]
        if live_held > 0 and zlib.crc32(Path(base).name.encode()) % live_held == 0:
            continue                                     # train_rec.py's live held-out: stays local
        files = [Path(base + s) for s in PARTS if Path(base + s).exists()]
        rel = [str(f.relative_to(ROOT)) for f in files]
        remote = remote_sha(rel)
        if any(remote.get(r) != sha(f) for r, f in zip(rel, files)):
            log(f"not on PBS yet (or differs), kept: {meta.name}")
            continue
        size = sum(f.stat().st_size for f in files)
        meta.unlink()                                    # .json first: an unfinished-looking run is skipped by readers
        for f in files:
            if f != meta:
                f.unlink()
        freed += size
    log(f"freed {freed / 1e9:.1f} GB of PBS-verified recordings; {free_gb():.0f} GB free")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=900)
    ap.add_argument("--low-gb", type=float, default=60.0)
    ap.add_argument("--free-gb", type=float, default=40.0)
    ap.add_argument("--live-held", type=int, default=10)
    ap.add_argument("--keep-after", default="2026-10-05 09:55", help="never prune recordings saved after this")
    a = ap.parse_args()
    subprocess.run(SSH + [f"mkdir -p {DEST}"], check=True)
    while True:
        t0 = time.time()
        ok = sync()
        log(f"sync {'ok' if ok else 'FAILED'} in {time.time() - t0:.0f} s; {free_gb():.0f} GB free")
        if ok and free_gb() < a.low_gb:
            prune(a.free_gb, a.live_held, time.mktime(time.strptime(a.keep_after, "%Y-%m-%d %H:%M")))
        time.sleep(max(60, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
