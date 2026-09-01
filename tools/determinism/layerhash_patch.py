"""Install forward hooks that hash every decoder layer's output, so an identical request can be
bisected: the first layer whose hash differs is where divergence originates.

Gated on VLLM_LAYER_HASH=1. Hashing forces a device sync per layer, so this is a diagnostic
build -- never leave it on for a benchmark. Use with --enforce-eager: hooks may not fire under
graph replay.
"""
import os, sys

TARGET = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/"
          "models/qwen3_8_flash_next/nvidia/model.py")

HOOK = '''

# ---- LAYER HASH DIAGNOSTIC (jschmied 2026-09-01), gated on VLLM_LAYER_HASH ----
# Hash every decoder layer's output so two identical requests can be bisected: the FIRST layer
# whose hash differs is where divergence originates. Installed by wrapping the class forward, so
# no insertion point inside __init__ is needed.
# Hashing forces a device sync per layer -- diagnostic only, never leave on for a benchmark.
# Use with --enforce-eager: hooks may not fire under graph replay.
def _install_layer_hashes(model):
    import hashlib, os, torch, logging
    if not os.environ.get("VLLM_LAYER_HASH"):
        return
    if getattr(model, "_layer_hash_installed", False):
        return
    model._layer_hash_installed = True
    _log = logging.getLogger("vllm")
    model._lh_pass = 0

    def _h(t):
        if not isinstance(t, torch.Tensor):
            return "n/a"
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]

    def _mk(name):
        def hook(_mod, _inp, out):
            o = out[0] if isinstance(out, (tuple, list)) and out else out
            h1 = _h(o)
            # v3: hash ONLY the first row too, and log the shape. GENBIS showed decode passes with
            # identical OUTPUT but differing full-tensor hashes at layers 1+: the hook was hashing
            # rows beyond the real token (num_actual_tokens=num_tokens_padded in the model runner).
            # row0 isolates the real token of a single-sequence decode.
            r0 = _h(o[:1]) if isinstance(o, torch.Tensor) and o.dim() >= 1 and o.shape[0] > 0 else "n/a"
            shp = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            _log.warning("LAYERHASH pass=%d %s %s row0=%s shape=%s",
                         getattr(model, "_lh_pass", -1), name, h1, r0, shp)
        return hook

    n = 0
    for name, mod in model.named_modules():
        if name.endswith(tuple("layers.%d" % i for i in range(64))) or "ple_embedding" in name:
            mod.register_forward_hook(_mk(name))
            n += 1
    _log.warning("LAYERHASH installed on %d modules", n)


_LH_ORIG_FORWARD = Qwen3_8FlashNextForConditionalGeneration.forward


def _lh_forward(self, *args, **kwargs):
    _install_layer_hashes(self)
    self._lh_pass = getattr(self, "_lh_pass", 0) + 1
    return _LH_ORIG_FORWARD(self, *args, **kwargs)


Qwen3_8FlashNextForConditionalGeneration.forward = _lh_forward
# ---- end LAYER HASH DIAGNOSTIC ----
'''

s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    i = s.find("\n\n# ---- LAYER HASH DIAGNOSTIC")
    if i < 0:
        print("  not installed"); raise SystemExit
    j = s.find("# ---- end LAYER HASH DIAGNOSTIC ----")
    s = s[:i] + s[j + len("# ---- end LAYER HASH DIAGNOSTIC ----"):]
    open(TARGET, "w").write(s); print("  layer-hash diagnostic REMOVED")
else:
    if "LAYER HASH DIAGNOSTIC" in s:
        print("  already installed"); raise SystemExit
    open(TARGET, "w").write(s + HOOK); print("  layer-hash diagnostic INSTALLED (inert unless VLLM_LAYER_HASH=1)")
