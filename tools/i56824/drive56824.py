#!/usr/bin/env python3
"""vllm#56824 repro on GB10 (root). Per arm: start the launcher as a transient unit, sample host memory every 0.5 s
(MemAvailable, the reporter's 'residual', swap) plus NVRM NV_ERR_NO_MEMORY lines from dmesg, and SIGKILL the unit
if MemAvailable stays under GUARD_GIB for two samples. Records the phase from the server log. Prod is stopped by the
job wrapper, not here.  argv: <out dir>"""
import json, os, re, subprocess, sys, time

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
LAUNCHER = "/opt/llm/serve-fnmain.sh"
VENV = "/opt/llm/runtime/vllm-venv-main1ea7"
MODEL = "/opt/llm/models/qwen38-flash-next-mtpfp4"
GUARD_GIB = 5.0
TIMEOUT_S = 2400
COMMON = {"FN_MAXLEN": "32768", "FN_UTIL": "0.70", "FN_SPEC_METHOD": "mtp", "FN_SPEC_N": "3",
          "FN_PLE_OFFLOAD": "0"}
MAPPED = '--engram-config {"checkpoint_mapped":true}'
REPRO_EXTRA = MAPPED + " --no-enable-prefix-caching --no-enable-flashinfer-autotune"
ARMS = [
    ("R0_control", {"FN_SEQS": "256", "FN_EXTRA": REPRO_EXTRA}),
    ("R1_repro", {"FN_SEQS": "256", "FN_KVDTYPE": "fp8_e4m3", "FN_CG_MODE": "FULL_DECODE_ONLY",
                  "FN_CG_SIZES": '[4,8,12,16,20,24],"mode":0', "FN_EXTRA": REPRO_EXTRA}),
    ("R2_repro_piecewise", {"FN_SEQS": "256", "FN_KVDTYPE": "fp8_e4m3", "FN_CG_MODE": "PIECEWISE",
                            "FN_CG_SIZES": '[4,8,12,16,20,24],"mode":0', "FN_EXTRA": REPRO_EXTRA}),
    ("R3_repro_seqs16", {"FN_SEQS": "16", "FN_KVDTYPE": "fp8_e4m3", "FN_CG_MODE": "FULL_DECODE_ONLY",
                         "FN_CG_SIZES": '[4,8,12,16,20,24],"mode":0', "FN_EXTRA": REPRO_EXTRA}),
]
PHASES = [("Loading weights took", "weights"), ("Model loading took", "model_loaded"),
          ("GPU KV cache size", "kv_sized"), ("Capturing CUDA graph", "capturing"),
          ("Graph capturing finished", "captured"), ("Application startup complete", "ready")]


def meminfo():
    m = {}
    for ln in open("/proc/meminfo"):
        k, v = ln.split(":"); m[k] = int(v.split()[0]) / 1048576
    m["residual"] = m["MemTotal"] - m["MemFree"] - m["Active"] - m["Inactive"] - m["Slab"] - m["PageTables"] - m["Unevictable"]
    return m


def nvrm_count():
    r = subprocess.run(["dmesg"], capture_output=True, text=True)
    return sum(1 for l in r.stdout.splitlines() if "NV_ERR_NO_MEMORY" in l)


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True)


def run_arm(name, env_over):
    unit = f"fx-i56824-{name.lower()}"
    lg = f"/opt/llm/i56824-{name}.log"
    sh("systemctl", "stop", unit); sh("systemctl", "reset-failed", unit)
    env = dict(COMMON); env.update(env_over)
    args = ["systemd-run", f"--unit={unit}", "--collect", "-E", f"FN_VENV={VENV}", "-E", f"FN_MODEL={MODEL}",
            "-E", f"FN_CACHE_ROOT=/opt/llm/.cache-i56824/{name}"]
    for k, v in env.items():
        args += ["-E", f"{k}={v}"]
    args += ["bash", "-c", f"{LAUNCHER} > {lg} 2>&1"]
    nv0 = nvrm_count()
    t0 = time.time(); subprocess.run(args, check=False)
    tsv = open(f"{OUT}/{name}.tsv", "w")
    tsv.write("t\tphase\tMemAvailable\tMemFree\tresidual\tSwapUsed\tCached\n")
    phase, low, outcome, minav, maxres, last_nv = "start", 0, None, 1e9, 0.0, time.time()
    nv = 0
    while time.time() - t0 < TIMEOUT_S:
        m = meminfo()
        try:
            txt = open(lg, errors="replace").read()
        except FileNotFoundError:
            txt = ""
        for pat, ph in PHASES:
            if pat in txt:
                phase = ph
        minav = min(minav, m["MemAvailable"]); maxres = max(maxres, m["residual"])
        tsv.write(f"{time.time()-t0:.1f}\t{phase}\t{m['MemAvailable']:.2f}\t{m['MemFree']:.2f}\t{m['residual']:.2f}\t"
                  f"{m['SwapTotal']-m['SwapFree']:.2f}\t{m['Cached']:.2f}\n")
        if time.time() - last_nv > 5:
            nv = nvrm_count() - nv0; last_nv = time.time()
        low = low + 1 if m["MemAvailable"] < GUARD_GIB else 0
        if low >= 2:
            sh("systemctl", "kill", "-s", "KILL", unit); outcome = f"GUARD_KILL at phase={phase} MemAvailable={m['MemAvailable']:.2f}"
            break
        if phase == "ready":
            outcome = "ready"; break
        if sh("systemctl", "is-active", unit).stdout.strip() not in ("active", "activating"):
            err = [l for l in txt.splitlines() if re.search(r"Error|error:|Traceback|OutOfMemory|out of memory", l)][-4:]
            outcome = "DIED at phase=" + phase + " | " + " || ".join(e.strip()[-220:] for e in err); break
        time.sleep(0.5)
    else:
        outcome = f"TIMEOUT at phase={phase}"
    tsv.close()
    serve_ok = None
    if outcome == "ready":
        r = sh("curl", "-s", "-m", "120", "http://127.0.0.1:8092/v1/chat/completions", "-H", "Authorization: Bearer sk-bench",
               "-H", "Content-Type: application/json", "-d",
               json.dumps({"model": "flashnext", "messages": [{"role": "user", "content": "Say hello in five words."}],
                           "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))
        serve_ok = '"choices"' in r.stdout
        time.sleep(3); m = meminfo(); minav = min(minav, m["MemAvailable"])
    nv = nvrm_count() - nv0
    kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", open(lg, errors="replace").read()) if os.path.exists(lg) else []
    sh("systemctl", "stop", unit)
    for _ in range(600):                              # let the pool drain before the next arm
        if meminfo()["MemAvailable"] >= 100: break
        time.sleep(1)
    res = dict(arm=name, outcome=outcome, served=serve_ok, min_MemAvailable_GiB=round(minav, 2),
               max_residual_GiB=round(maxres, 2), nvrm_no_memory_lines=nv, kv_tokens=kv[-1] if kv else None,
               secs=round(time.time() - t0))
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    print(f"baseline meminfo: {json.dumps({k: round(v, 2) for k, v in meminfo().items() if k in ('MemAvailable', 'MemFree', 'residual', 'SwapTotal', 'SwapFree')})}", flush=True)
    sel = [a for a in os.environ.get("ARMS_SEL", "").split(",") if a]
    for name, env in ARMS:
        if not sel or name in sel:
            run_arm(name, env)
    print("== ALL DONE ==", flush=True)
