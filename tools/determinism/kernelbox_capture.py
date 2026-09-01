#!/usr/bin/env python3
"""Install/remove capturing wrappers around Flash-Next's registered custom ops.

Treats each kernel as a BOX: record everything crossing its boundary during a REAL request
(synthetic inputs may never hit the triggering shapes/values), so it can be re-run standalone.
Two properties then become testable offline, in a fresh process:

  determinism  replay identical captured inputs N times -> outputs must be bit-identical
  purity       replay(captured_inputs) must equal the captured output. If not, the kernel
               depends on something not in its inputs -- stale memory, a global, workspace
               state -- and is not a function at all.

MUTATING OPS matter here: topk writes into `blocks`, qsa_with_output writes its output arg.
So args are cloned BEFORE the call (pre-image) and re-read AFTER (post-image), and replay
restores the pre-image every iteration or it measures the previous iteration's leftovers.

Usage:  python kernelbox_capture.py [off]
Gated at runtime on VLLM_KERNEL_CAPTURE=<dir>; inert without it. Diagnostic only -- cloning
every arg forces a sync per call.
"""
import os, sys

TARGET = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/"
          "models/qwen3_8_flash_next/nvidia/model.py")

BEGIN = "# ---- KERNEL BOX CAPTURE"
END = "# ---- end KERNEL BOX CAPTURE ----"

HOOK = '''

# ---- KERNEL BOX CAPTURE (jschmied 2026-09-01), gated on VLLM_KERNEL_CAPTURE ----
def _install_kernel_capture():
    import os, torch, logging
    d = os.environ.get("VLLM_KERNEL_CAPTURE")
    if not d or getattr(torch, "_kbox_installed", False):
        return
    torch._kbox_installed = True
    os.makedirs(d, exist_ok=True)
    _log = logging.getLogger("vllm")
    SKIP = int(os.environ.get("VLLM_KERNEL_CAPTURE_SKIP", "4"))
    KEEP = int(os.environ.get("VLLM_KERNEL_CAPTURE_KEEP", "2"))
    want = [s.strip() for s in os.environ.get(
        "VLLM_KERNEL_CAPTURE_OPS",
        "_C.persistent_topk,_C.cooperative_topk,"
        "vllm.qwen3_8_flash_next_qsa_with_output,"
        "vllm.qwen3_8_flash_next_ple_short_conv,"
        "vllm.qwen3_8_flash_next_low_latency_gemm").split(",") if s.strip()]
    counts = {}

    def _snap(x):
        if isinstance(x, torch.Tensor):
            return x.detach().clone()
        if isinstance(x, (list, tuple)):
            return type(x)(_snap(i) for i in x)
        if isinstance(x, dict):
            return {k: _snap(v) for k, v in x.items()}
        return x

    def _wrap(key, orig):
        def w(*args, **kwargs):
            n = counts.get(key, 0)
            counts[key] = n + 1
            grab = SKIP <= n < SKIP + KEEP
            if not grab:
                return orig(*args, **kwargs)
            pre, prekw = _snap(args), _snap(kwargs)
            out = orig(*args, **kwargs)
            torch.cuda.synchronize()
            torch.save({"op": key, "call": n,
                        "args_pre": pre, "kwargs_pre": prekw,
                        "args_post": _snap(args), "kwargs_post": _snap(kwargs),
                        "ret": _snap(out)},
                       os.path.join(d, key.replace(".", "_") + "__%d.pt" % n))
            _log.warning("KBOX captured %s call=%d", key, n)
            return out
        return w

    ok = []
    for spec in want:
        nsname, _, opname = spec.partition(".")
        try:
            ns = getattr(torch.ops, nsname)
            orig = getattr(ns, opname)
        except Exception as e:
            _log.warning("KBOX cannot resolve %s: %s", spec, e)
            continue
        setattr(ns, opname, _wrap(spec, orig))
        ok.append(spec)
    _log.warning("KBOX armed on %d/%d ops: %s", len(ok), len(want), ",".join(ok))


try:
    _install_kernel_capture()
except Exception as _e:
    import logging
    logging.getLogger("vllm").warning("KBOX install failed: %s", _e)
# ---- end KERNEL BOX CAPTURE ----
'''

s = open(TARGET).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if BEGIN in s:
        s = s[:s.index("\n\n" + BEGIN)] + "\n"
        open(TARGET, "w").write(s)
        print("  kernel-box capture REMOVED")
    else:
        print("  kernel-box capture was not installed")
else:
    if BEGIN in s:
        print("  kernel-box capture already installed")
    else:
        open(TARGET, "w").write(s + HOOK)
        print("  kernel-box capture INSTALLED (inert unless VLLM_KERNEL_CAPTURE=<dir>)")
