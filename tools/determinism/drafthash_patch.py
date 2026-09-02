"""Hash the DRAFTER's input and every submodule output at the first calls of each turn, so a
healthy and a broken draw of the same prompt can be bisected inside the MTP model.
Gated on VLLM_DRAFT_HASH=1; needs --enforce-eager (hooks do not fire under compile/graphs)."""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/models/qwen3_8_flash_next/nvidia/mtp.py"
HOOK = '''

# ---- DRAFT HASH DIAGNOSTIC (jschmied 2026-09-02), gated on VLLM_DRAFT_HASH ----
def _install_draft_hashes(wrapper):
    import hashlib, os, torch, logging
    if not os.environ.get("VLLM_DRAFT_HASH"):
        return False
    if getattr(wrapper, "_dh_installed", False):
        return True
    wrapper._dh_installed = True
    _log = logging.getLogger("vllm")
    wrapper._dh_turn = 0; wrapper._dh_call = 0; wrapper._dh_budget = 0

    def _h(t):
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return "n/a"
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]
    wrapper._dh_h = _h

    def _mk(name):
        def hook(_mod, _inp, out):
            if wrapper._dh_budget <= 0:
                return
            o = out[0] if isinstance(out, (tuple, list)) and out else out
            r0 = _h(o[:1]) if isinstance(o, torch.Tensor) and o.dim() >= 1 else "n/a"
            shp = tuple(o.shape) if isinstance(o, torch.Tensor) else None
            _log.warning("DRAFTHASH turn=%d call=%d %s row0=%s shape=%s",
                         wrapper._dh_turn, wrapper._dh_call, name, r0, shp)
        return hook

    n = 0
    for name, mod in wrapper.model.named_modules():
        if not name:
            continue
        depth = name.count(".")
        if depth <= 1 or name.startswith("layers.") and depth <= 3:
            mod.register_forward_hook(_mk(name)); n += 1
    _log.warning("DRAFTHASH installed on %d modules", n)
    return True


_DH_ORIG_FORWARD = Qwen3_8FlashNextMTP.forward


def _dh_forward(self, input_ids, positions, hidden_states=None, *args, **kwargs):
    if _install_draft_hashes(self):
        import logging
        _log = logging.getLogger("vllm")
        nt = int(hidden_states.shape[0]) if hidden_states is not None else -1
        if nt > 16:                       # a draft PREFILL = a new turn: open a budget of 7 calls
            self._dh_turn += 1; self._dh_call = 0; self._dh_budget = 7
        if self._dh_budget > 0:
            self._dh_call += 1
            _log.warning("DRAFTHASH turn=%d call=%d INPUT ntok=%d ids=%s pos=%s hid_row0=%s hid_rowN=%s",
                         self._dh_turn, self._dh_call, nt,
                         input_ids[:3].tolist() if input_ids is not None else None,
                         positions[:1].tolist(), self._dh_h(hidden_states[:1]), self._dh_h(hidden_states[-1:]))
        out = _DH_ORIG_FORWARD(self, input_ids, positions, hidden_states, *args, **kwargs)
        if self._dh_budget > 0:
            o = out[0] if isinstance(out, tuple) else out
            _log.warning("DRAFTHASH turn=%d call=%d OUTPUT row0=%s", self._dh_turn, self._dh_call, self._dh_h(o[:1]))
            self._dh_budget -= 1
        return out
    return _DH_ORIG_FORWARD(self, input_ids, positions, hidden_states, *args, **kwargs)


Qwen3_8FlashNextMTP.forward = _dh_forward
# ---- end DRAFT HASH DIAGNOSTIC ----
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    i = s.find("\n\n# ---- DRAFT HASH DIAGNOSTIC")
    if i < 0: print("  not installed"); raise SystemExit
    j = s.find("# ---- end DRAFT HASH DIAGNOSTIC ----")
    rest = s[j + len("# ---- end DRAFT HASH DIAGNOSTIC ----"):]
    open(TARGET, "w").write(s[:i] + (rest[1:] if rest.startswith("\n") else rest)); print("  draft-hash diagnostic REMOVED")
else:
    if "DRAFT HASH DIAGNOSTIC" in s: print("  already installed"); raise SystemExit
    open(TARGET, "w").write(s + HOOK); print("  draft-hash diagnostic INSTALLED (inert unless VLLM_DRAFT_HASH=1)")
