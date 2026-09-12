#!/usr/bin/env python3
"""qnext -- advance the work queue by exactly ONE job, then stop.

Not an autonomous scheduler. The queue was never what failed; the timing was. Read-out crons were
set on a wall clock and guessed wrong twice (fired before a run finished), while three fired after
the work was already done. So this replaces "guess when it ends" with "watch for the sentinel", and
keeps a checkpoint between every job rather than running the queue unattended.

  qnext.py --status              show the queue, start nothing
  qnext.py --dry-run             show what WOULD start, and why the others were skipped
  qnext.py                       claim one runnable job, run it to its sentinel, record, exit

Rules it enforces:
  * refuses while any fx-* unit is live, except *-dl (downloads are not GPU work)
  * never starts an item marked needs_user -- prod changes, downloads, upstream posts
  * an item runs only when every prereq is `done`
  * one job per invocation; the watchdog tick calls it again
  * a lock file, so two invocations cannot claim the same job
  * completion is the job's own `== ALL DONE ==` / `== VOID ==` sentinel, not elapsed time; a job
    that exits without either is recorded `unknown`, never silently `done`

Queue item:
  {"name":"cb3-tier0", "cmd":["python3","/opt/llm/runners/cb3_random.py"],
   "needs_user":false, "prereqs":[], "expect_min":15, "note":"what it decides"}
"""
from __future__ import annotations
import argparse, fcntl, json, os, subprocess, sys, time

QUEUE = os.environ.get("QNEXT_QUEUE", "/opt/llm/qnext/queue.json")
LOCK = "/opt/llm/qnext/.lock"
LOGDIR = "/opt/llm/qnext/logs"
DONE, VOID = "== ALL DONE ==", "== VOID =="


def log(*a): print(*a, flush=True)
def load(): return json.load(open(QUEUE))
def missing_artifacts(job: dict) -> list[str]:
    """File arguments in a job's cmd that do not exist.

    A job whose script or spec is absent is not work, it is a scheduled failure -- and it will fire
    at whatever hour the chain reaches it. pr56509 sat queued for hours with no script (2026-09-12).
    qnext skips such a job and moves on rather than burning the slot.
    """
    out = []
    for a in job.get("cmd", []):
        if "/" in a and a.rsplit(".", 1)[-1] in ("py", "sh", "json") and not os.path.exists(a):
            out.append(a)
    return out


def save(q):
    """Atomic, and ownership-preserving: qnext runs as root for jobs that touch /opt/llm/runtime,
    and a root-owned queue silently blocks the next non-root --status/edit."""
    tmp = QUEUE + ".tmp"
    try:
        st = os.stat(QUEUE)
    except FileNotFoundError:
        st = None
    json.dump(q, open(tmp, "w"), indent=1)
    os.replace(tmp, QUEUE)
    if st and os.geteuid() == 0:
        os.chown(QUEUE, st.st_uid, st.st_gid)


def busy() -> list[str]:
    out = subprocess.run(["systemctl", "list-units", "--no-legend", "fx-*"],
                         capture_output=True, text=True).stdout
    return [l.split()[0] for l in out.splitlines()
            if l.strip() and not l.split()[0].startswith("fx-qnext")
            and "-dl." not in l.split()[0]]


def runnable(q):
    done = {i["name"] for i in q["items"] if i.get("state") == "done"}
    out = []
    for i in q["items"]:
        if i.get("state", "queued") != "queued":
            out.append((i, f"state={i.get('state')}")); continue
        if i.get("needs_user"):
            out.append((i, "needs_user")); continue
        absent = missing_artifacts(i)
        if absent:
            # A job whose script/spec does not exist is a scheduled failure, not work.
            out.append((i, "NO ARTIFACT: " + ", ".join(os.path.basename(a) for a in absent)))
            continue
        missing = [p for p in i.get("prereqs", []) if p not in done]
        out.append((i, f"waiting on {','.join(missing)}" if missing else None))
    return out


def show(q, dry):
    log(f"{'job':<24} {'state':<9} {'min':>4}  why not / note")
    seen_next = False
    for i, why in runnable(q):
        if why is None and not seen_next:
            tag, seen_next = "RUNNABLE <- next", True
        elif why is None:
            tag = "runnable"
        else:
            tag = why
        log(f"  {i['name']:<22} {i.get('state','queued'):<9} {i.get('expect_min','?'):>4}  {tag}")
    nxt = next((i for i, w in runnable(q) if w is None), None)
    log(f"\n  next: {nxt['name'] if nxt else '(nothing runnable)'}")
    if nxt and dry:
        log(f"  would run: {' '.join(nxt['cmd'])}")
    b = busy()
    if b: log(f"  BUT the box is busy: {', '.join(b)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    q = load()
    if a.status or a.dry_run:
        show(q, a.dry_run); return 0

    b = busy()
    if b:
        log(f"REFUSING: fx-* active: {', '.join(b)}"); return 3

    os.makedirs(LOGDIR, exist_ok=True)
    lf = open(LOCK, "w")
    try:
        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("REFUSING: another qnext holds the lock"); return 3

    q = load()
    job = next((i for i, w in runnable(q) if w is None), None)
    if not job:
        log("nothing runnable"); return 0

    job["state"] = "running"; job["started"] = time.strftime("%F %T")
    lg = f"{LOGDIR}/{job['name']}.log"; job["log"] = lg
    save(q)
    log(f"== qnext: {job['name']} ==  {job.get('note','')}")
    log(f"   {' '.join(job['cmd'])}\n   log {lg}")

    t0 = time.time()
    with open(lg, "w") as fh:
        rc = subprocess.run(job["cmd"], stdout=fh, stderr=subprocess.STDOUT).returncode
    mins = (time.time() - t0) / 60
    text = open(lg, errors="replace").read()
    state = "done" if DONE in text else "void" if VOID in text else "unknown"

    q = load()
    for i in q["items"]:
        if i["name"] == job["name"]:
            i.update(state=state, finished=time.strftime("%F %T"), rc=rc, minutes=round(mins, 1))
    save(q)
    log(f"\n== {job['name']}: {state} (rc={rc}, {mins:.0f} min) ==")
    if state == "unknown":
        log("   no sentinel in the output -- NOT recorded as done. Read the log before trusting it.")
    log(f"   tail: {text.strip().splitlines()[-1][:110] if text.strip() else '(empty)'}")
    nxt = next((i for i, w in runnable(load()) if w is None), None)
    log(f"   next runnable: {nxt['name'] if nxt else '(nothing)'}")
    return 0 if state == "done" else 2


if __name__ == "__main__":
    sys.exit(main())
