#!/usr/bin/env python3
"""armrun -- declarative server A/B on the GB10. One spec file, arms as data, void checks enforced.

Replaces 13 hand-copied drivers (~1,500 lines) that all did the same thing: start a vLLM serve unit
with FN_* env, wait for startup, assert something about the log, run a probe, report per arm. Copying
them produced five defects in one day, none interesting:

  * a probe pointed at port 8080 while the launcher serves 8092            -> PORT is a constant here
  * a void marker that also matched when the patch was ABSENT, so both
    arms reported "installed" and the A/B would have compared prod to prod -> markers are validated
  * sys.exit on VOID skipped cleanup and orphaned a running server          -> cleanup is in finally
  * a control arm forced a flag to "1", taking a different code branch
    than the default it was meant to reproduce                              -> env_unset, not env=1
  * a docstring splice left a syntax error in the installed copy            -> no more copying

Spec (JSON):
{
  "name": "det-221-compile",
  "venv": "/opt/llm/runtime/vllm-venv-fnmain3",
  "model": "/opt/llm/models/qwen38-flash-next-nvfp4",
  "env":  {"FN_MAXLEN":"8192","FN_SEQS":"4","FN_UTIL":"0.75","FN_MTP":"3"},
  "probe_cmd": ["<venv>/bin/python","/opt/llm/runners/hiprobe_probe.py","--url","{url}","--arm","{arm}"],
  "starts": 1,
  "arms": [
    {"name":"cc_on",  "log_must_contain":["torch.compile took"], "expect_wrong":["hi-08","hi-11"]},
    {"name":"cc_off", "env":{"FN_CG_SIZES":"[1,2,4,8],\\"mode\\":0"},
                      "log_must_contain":["CompilationMode.NONE"],
                      "log_must_not_contain":["torch.compile took"]},
    {"name":"patched","source_toggle":{"script":"/opt/llm/runners/gdn55715_patch.py",
                                       "file":"<pkg>/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
                                       "marker":"GDN55715-PATCH","on":true}}
  ]
}

An arm's `env` is merged over the spec `env`. `env_unset` removes keys, for a control that must take
the *default* branch rather than an explicitly-set one. `probe_cmd` must print ONE json object on
stdout; `{"wrong": [...], "n": 12}` enables the `expect_wrong` control check, anything else is just
recorded.

Exit codes: 0 all arms ran and every declared check held; 2 VOID (a check failed -- numbers from that
arm must not be used); 3 setup refusal (something else is on the GPU).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

PORT = 8092                      # the launcher serves this. Not a default copied from elsewhere.
URL = f"http://127.0.0.1:{PORT}"
LAUNCHER = "/opt/llm/serve-fnmain.sh"
STARTUP_SENTINEL = "Application startup complete"
STARTUP_TIMEOUT_S = 2400         # ~12 min is normal for this model; 131072 ctx is slower


def log(*a): print(*a, flush=True)
def sh(*c): return subprocess.run(c, capture_output=True, text=True)


def refuse_if_busy(unit: str) -> None:
    out = sh("systemctl", "list-units", "--no-legend", "fx-*").stdout
    # fx-qnext is the orchestrator that launched us, not a GPU job; downloads are not GPU work.
    # Without this, qnext running as fx-qnext makes every job it starts refuse itself (rc=3).
    others = [l.split()[0] for l in out.splitlines()
              if l.strip() and not l.split()[0].startswith((unit, f"{unit}-", "fx-qnext"))
              and "-dl." not in l.split()[0]]
    if others:
        log(f"!! REFUSING: other fx-* units active: {', '.join(others)}")
        sys.exit(3)


def pkg_dir(venv: str) -> str:
    return f"{venv}/lib/python3.12/site-packages/vllm"


def toggle(spec_venv: str, t: dict, on: bool) -> bool:
    """Apply/remove a source patch and return the state READ BACK from the file."""
    f = t["file"].replace("<pkg>", pkg_dir(spec_venv))
    if not os.access(f, os.W_OK):
        # Found by armrun's own first test run: as a non-root user the patch script cannot
        # write /opt/llm/runtime/..., fails quietly, and BOTH states then read identical --
        # which the marker check reports as "not discriminating", blaming the wrong thing.
        log(f"!! REFUSING: {f} is not writable as {os.environ.get('USER','?')}. "
            f"Source toggles need root -- run armrun under systemd-run as root.")
        sys.exit(3)
    env = dict(os.environ)
    for k in ("VLLM_PKG", "GDN_PY", "VLLM_CONN_PY", "VLLM_QSA_PY", "VLLM_FCM_PY", "VLLM_FI_CORE_PY"):
        env.setdefault(k, f)
    subprocess.run([f"{spec_venv}/bin/python", t["script"]] + ([] if on else ["off"]),
                   capture_output=True, text=True, env=env)
    try:
        return t["marker"] in open(f, encoding="utf8").read()
    except FileNotFoundError:
        return False


def validate_marker(spec_venv: str, t: dict) -> None:
    """A marker that matches in BOTH states cannot witness anything. Refuse such a spec."""
    on = toggle(spec_venv, t, True)
    off = toggle(spec_venv, t, False)
    toggle(spec_venv, t, bool(t.get("on", True)))          # leave as declared
    if not (on and not off):
        log(f"!! REFUSING: marker {t['marker']!r} reads on={on} off={off} -- not discriminating "
            f"(the file IS writable, so this is the marker, not permissions). "
            f"Pick a string unique to the patch body (this exact trap voided a run on 2026-09-11).")
        sys.exit(3)


def start(unit: str, venv: str, model: str, env: dict, tag: str) -> tuple[str | None, str]:
    lg = f"/opt/llm/armrun-{tag}.log"
    sh("systemctl", "stop", unit); sh("systemctl", "reset-failed", unit)
    args = ["systemd-run", f"--unit={unit}", "--collect",
            "-E", f"FN_VENV={venv}", "-E", f"FN_MODEL={model}",
            "-E", f"FN_CACHE_ROOT=/opt/llm/.cache-armrun/{tag}"]
    for k, v in env.items():
        args += ["-E", f"{k}={v}"]
    args += ["bash", "-c", f"{LAUNCHER} > {lg} 2>&1"]
    subprocess.run(args, check=False)
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        time.sleep(10)
        try:
            t = open(lg, errors="replace").read()
        except FileNotFoundError:
            continue
        if STARTUP_SENTINEL in t:
            return lg, t
        for fatal in ("Failed core proc", "CUDA out of memory", "Engine core initialization failed"):
            if fatal in t:
                return None, t
    return None, ""


def check_log(arm: dict, venv: str, text: str) -> list[str]:
    fails = []
    if f"vllm-venv-{venv.rstrip('/').split('-')[-1]}" not in text and venv.rstrip("/").split("/")[-1] not in text:
        fails.append(f"log does not name the venv {venv.rstrip('/').split('/')[-1]}")
    for s in arm.get("log_must_contain", []):
        if s not in text: fails.append(f"log missing required {s!r}")
    for s in arm.get("log_must_not_contain", []):
        if s in text: fails.append(f"log contains forbidden {s!r}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec"); ap.add_argument("--out", default=None)
    ap.add_argument("--starts", type=int, default=None)
    ap.add_argument("--validate-only", action="store_true",
                    help="check the spec (busy box, writability, discriminating markers) and exit "
                         "WITHOUT starting a server. Added after a marker test launched a real one.")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    name = spec["name"]; unit = f"fx-armrun-{name}"
    venv = spec["venv"]; model = spec["model"]
    starts = a.starts or spec.get("starts", 1)
    out = a.out or f"/opt/llm/runners/results/armrun-{name}.jsonl"
    refuse_if_busy(unit)

    for arm in spec["arms"]:
        if "source_toggle" in arm:
            validate_marker(venv, arm["source_toggle"])
    if a.validate_only:
        log(f"== armrun {name}: spec valid ({len(spec['arms'])} arms). Nothing started. ==")
        globals()["_SENT"] = True      # a dry check is not a completed job
        return 0

    log(f"== armrun {name}: {len(spec['arms'])} arms x {starts} start(s), port {PORT} ==")
    res: dict[str, list] = {arm["name"]: [] for arm in spec["arms"]}
    rc = 0
    fh = open(out, "a")
    try:
        for i in range(starts):
            for arm in spec["arms"]:
                tag = f"{name}-{arm['name']}{i}"
                env = dict(spec.get("env", {})); env.update(arm.get("env", {}))
                for k in arm.get("env_unset", []): env.pop(k, None)
                st = arm.get("source_toggle")
                if st:
                    got = toggle(venv, st, bool(st.get("on", True)))
                    if got != bool(st.get("on", True)):
                        log(f"!! VOID {tag}: toggle state {got}, wanted {st.get('on', True)}"); return 2
                log(f"-- {tag}  env={ {k: v for k, v in arm.get('env', {}).items()} or '(spec default)'}")
                lg, text = start(unit, venv, model, env, tag)
                if not lg:
                    log(f"!! VOID {tag}: server never reached startup (log {lg or '/opt/llm/armrun-'+tag+'.log'})")
                    return 2
                fails = check_log(arm, venv, text)
                if fails:
                    for f in fails: log(f"!! VOID {tag}: {f}")
                    return 2
                log(f"   void checks passed ({len(arm.get('log_must_contain', []))} required, "
                    f"{len(arm.get('log_must_not_contain', []))} forbidden)")
                cmd = [c.replace("{url}", URL).replace("{arm}", tag).replace("<venv>", venv)
                       for c in spec["probe_cmd"]]
                p = sh(*cmd)
                try:
                    m = json.loads([l for l in p.stdout.splitlines() if l.strip().startswith("{")][-1])
                except Exception:
                    log(f"!! VOID {tag}: probe printed no json object\n{p.stdout[-600:]}{p.stderr[-400:]}")
                    return 2
                m.update(arm=arm["name"], start=i, tag=tag, ts=time.strftime("%F %T"))
                fh.write(json.dumps(m, ensure_ascii=False) + "\n"); fh.flush()
                res[arm["name"]].append(m)
                log(f"   {json.dumps({k: v for k, v in m.items() if k not in ('arm','start','tag','ts')})}")
                exp = arm.get("expect_wrong")
                if exp is not None and sorted(m.get("wrong", [])) != sorted(exp):
                    log(f"   !! CONTROL MISMATCH: expected wrong={sorted(exp)}, got={sorted(m.get('wrong', []))}")
                    log( "   !! a control that does not reproduce invalidates the other arms -- explain this first")
                    rc = 2
    finally:
        sh("systemctl", "stop", unit)
        for arm in spec["arms"]:                       # restore declared source state
            if "source_toggle" in arm:
                st = arm["source_toggle"]; toggle(venv, st, bool(st.get("restore", st.get("on", True))))
        fh.close()

    log("\n== RESULT ==")
    for arm in spec["arms"]:
        rows = res[arm["name"]]
        if not rows: continue
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
        parts = []
        for k in keys:
            vs = [r[k] for r in rows]
            parts.append(f"{k} {min(vs)}" if min(vs) == max(vs) else f"{k} {min(vs)}–{max(vs)}")
        w = sorted({x for r in rows for x in r.get("wrong", [])})
        log(f"  {arm['name']:<12} {'  '.join(parts)}" + (f"   wrong: {','.join(w)}" if w else ""))
    log("  ranges, not means -- reps within a start are not exchangeable")
    log("== ALL DONE ==" if rc == 0 else "== VOID ==")
    globals()["_SENT"] = True
    return rc


if __name__ == "__main__":
    # Early `return 2` paths (VOID) skip the summary block, so the sentinel never printed and
    # qnext recorded the job `unknown` instead of `void`. 2026-09-12.
    _rc = main()
    if not globals().get("_SENT"):
        log("== VOID ==" if _rc else "== ALL DONE ==")
    sys.exit(_rc)
