#!/usr/bin/env python3
"""Hash the GDN recurrent state (conv + SSM) at this request's slot after every GDN forward.

Why: with the prefix cache off and speculation off, the prefill's OUTPUT is bit-identical across
requests but decode step 1 already diverges (finding 21, n=2). The one thing carried from the
identical prefill into the diverging decode is the recurrent state written at
`ssm_state[prefill_state_indices] = last_recurrent_state` (qwen_gdn_linear_attn.py:1520) and read
back at decode. Hashing that state at the slot, right after each forward, splits the question:

  prefill-pass state identical x3, decode diverges  -> the WRITE is deterministic; the fault is in
                                                       decode's read/compute of it
  prefill-pass state DIFFERS x3 (output identical!) -> the final-state write is nondeterministic
                                                       even when the prefill logits are not

Logs: STATEHASH pass=N layer=<prefix> kind=<prefill|decode|spec> slots=[...] ssm=<h> conv=<h>
Layers 0 and 1 only (both linear_attention; 1 also carries the PLE). Gated on VLLM_STATE_HASH=1.
Usage: statehash_patch.py [off]
"""
import sys
T = ("/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/"
     "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py")
BEGIN = "# ---- GDN STATE HASH DIAGNOSTIC"
HOOK = '''

# ---- GDN STATE HASH DIAGNOSTIC (jschmied 2026-09-02), gated on VLLM_STATE_HASH ----
def _sh_install():
    import os, hashlib, logging, torch
    if not os.environ.get("VLLM_STATE_HASH"):
        return
    _log = logging.getLogger("vllm")
    _orig = QwenGatedDeltaNetAttention.forward
    _pass = {"n": 0}

    def _h(t):
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]

    def _fwd(self, *a, **kw):
        out = _orig(self, *a, **kw)
        try:
            pfx = getattr(self, "prefix", "?")
            if ".layers.0." not in pfx and ".layers.1." not in pfx:
                return out
            if ".layers.0." in pfx:
                _pass["n"] += 1
            md = get_forward_context().attn_metadata
            if isinstance(md, dict):
                md = md.get(pfx)
            kind, idx = None, None
            for name, k in (("prefill_state_indices", "prefill"),
                            ("non_spec_state_indices_tensor", "decode"),
                            ("spec_state_indices_tensor", "spec")):
                t = getattr(md, name, None) if md is not None else None
                if t is not None and t.numel() > 0:
                    v = t.reshape(-1)
                    v = v[v >= 0]
                    if v.numel() > 0:
                        kind, idx = k, v
                        break
            if idx is None:
                _log.warning("STATEHASH pass=%d layer=%s kind=none (no slot in metadata)", _pass["n"], pfx)
                return out
            torch.cuda.synchronize()
            conv, ssm = self.kv_cache[0], self.kv_cache[1]
            _log.warning("STATEHASH pass=%d layer=%s kind=%s slots=%s ssm=%s conv=%s ssm_shape=%s",
                         _pass["n"], pfx, kind, idx.tolist()[:4], _h(ssm[idx]), _h(conv[idx]),
                         tuple(ssm.shape[1:]))
        except Exception as e:  # never break the forward
            _log.warning("STATEHASH error: %r", e)
        return out

    QwenGatedDeltaNetAttention.forward = _fwd
    _log.warning("STATEHASH armed on layers 0 and 1")


try:
    _sh_install()
except Exception as _e:
    import logging
    logging.getLogger("vllm").warning("STATEHASH install failed: %r", _e)
# ---- end GDN STATE HASH DIAGNOSTIC ----
'''
s = open(T).read()
if len(sys.argv) > 1 and sys.argv[1] == "off":
    if BEGIN in s:
        open(T, "w").write(s[:s.index("\n\n" + BEGIN)] + "\n"); print("  state-hash REMOVED")
    else:
        print("  state-hash not installed")
else:
    if BEGIN in s:
        print("  state-hash already installed")
    else:
        assert "class QwenGatedDeltaNetAttention" in s and "get_forward_context" in s
        open(T, "w").write(s + HOOK); print("  state-hash INSTALLED (inert unless VLLM_STATE_HASH=1)")
