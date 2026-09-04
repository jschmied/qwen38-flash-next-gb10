"""main-tree port: pass quant_config to the body ParallelLMHead (FP8 lm_head checkpoints ship
`lm_head.weight_scale`; main constructs the head unquantized -> "no parameter named lm_head.weight_scale").
Target: vllm/models/qwen4_exp/nvidia/mtp.py (the MTP head builds its own ParallelLMHead the same way and the
checkpoint's `lm_head.weight_scale_inv` maps onto it via `_map_mtp_name` -> same failure with MTP on). `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_MTP_PY"): return os.environ["VLLM_MTP_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "models/qwen4_exp/nvidia/mtp.py")
TARGET = _target()
ANCHOR = '''                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
'''
NEW = '''                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=vllm_config.quant_config,  # LMHEADQ-MTP (jschmied): FP8 lm_head checkpoints
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  lmheadq-mtp not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  lmheadq-mtp REMOVED")
else:
    if "LMHEADQ" in s: print("  lmheadq already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  lmheadq-mtp INSTALLED in", TARGET)
