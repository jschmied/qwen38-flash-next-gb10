#!/usr/bin/env python3
"""Hash the PLE layer's inputs, output and conv state on every call.

Finding 24: layer 0's GDN state is deterministic through decode, layer 1's diverges at decode
step 1, and the PLE is the only structural difference. The PLE is (a) an offloaded n-gram table
lookup in a subprocess with an async connector and (b) a MambaBase with its own short-conv state.
This hook separates "the PLE receives different inputs" from "the PLE computes differently":

  inputs identical, output DIFFERS  -> the PLE internals (offload lookup / short conv / its state)
  inputs already differ             -> divergence enters upstream of the PLE at decode step 1
  output identical, conv state differs -> the state write, not the returned tensor

Output is hashed twice with a sync between (RACE flag): an async copy landing after the return
would otherwise look "identical" here and different at the consumer.
Logs: PLEHASH pass=N in_hs=.. in_ids=.. in_ctx=.. out=..[ RACE h2=..] convstate=.. shape=..
Gated on VLLM_PLE_HASH=1. Usage: plehash_patch.py [off]
"""
import sys
T = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/"
     "models/qwen3_8_flash_next/nvidia/ple_layer.py")
BEGIN = "# ---- PLE HASH DIAGNOSTIC"
HOOK = '''

# ---- PLE HASH DIAGNOSTIC (jschmied 2026-09-02), gated on VLLM_PLE_HASH ----
def _ph_install():
    import os, hashlib, logging, torch
    if not os.environ.get("VLLM_PLE_HASH"):
        return
    _log = logging.getLogger("vllm")
    _orig = Qwen3_8FlashNextPLELayer.forward
    _n = {"p": 0}

    def _h(t):
        if not isinstance(t, torch.Tensor) or t.numel() == 0:
            return "n/a"
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]

    def _fwd(self, hidden_states, input_ids, query_start_loc, ngram_context, *a, **kw):
        _n["p"] += 1
        try:
            torch.cuda.synchronize()
            ih, ii, ic = _h(hidden_states), _h(input_ids), _h(ngram_context)
        except Exception as e:
            ih = ii = ic = "err:%r" % (e,)
        out = _orig(self, hidden_states, input_ids, query_start_loc, ngram_context, *a, **kw)
        try:
            torch.cuda.synchronize()
            h1 = _h(out)
            torch.cuda.synchronize()
            h2 = _h(out)
            cs = self.kv_cache[0] if isinstance(self.kv_cache, (tuple, list)) else self.kv_cache
            _log.warning("PLEHASH pass=%d in_hs=%s in_ids=%s in_ctx=%s out=%s%s convstate=%s shape=%s",
                         _n["p"], ih, ii, ic, h1, "" if h1 == h2 else "  RACE h2=" + h2,
                         _h(cs), tuple(hidden_states.shape))
        except Exception as e:
            _log.warning("PLEHASH error: %r", e)
        return out

    Qwen3_8FlashNextPLELayer.forward = _fwd
    _log.warning("PLEHASH armed")


try:
    _ph_install()
except Exception as _e:
    import logging
    logging.getLogger("vllm").warning("PLEHASH install failed: %r", _e)
# ---- end PLE HASH DIAGNOSTIC ----
'''
s = open(T).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if BEGIN in s:
        open(T, "w").write(s[:s.index("\n\n" + BEGIN)] + "\n"); print("  ple-hash REMOVED")
    else:
        print("  ple-hash not installed")
else:
    if BEGIN in s:
        print("  ple-hash already installed")
    else:
        assert "class Qwen3_8FlashNextPLELayer" in s
        open(T, "w").write(s + HOOK); print("  ple-hash INSTALLED (inert unless VLLM_PLE_HASH=1)")
