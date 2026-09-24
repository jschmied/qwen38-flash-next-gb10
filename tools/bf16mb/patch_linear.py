import sys
p = sys.argv[1]; s = open(p).read()
MARK = "FNBF16SK"
if MARK in s:
    print("already patched"); sys.exit(0)
anchor = "class UnquantizedLinearMethod(LinearMethodBase):"
assert s.count(anchor) == 1
s = s.replace(anchor, '''# ---- FNBF16SK (jschmied 2026-09-25, local overlay): FN_BF16SK=1 routes small-M BF16 linears to a deterministic
# split-K Triton GEMM (model_executor/layers/fn_bf16sk.py); off = stock path, untouched ----
_FN_BF16SK = __import__("os").environ.get("FN_BF16SK", "") == "1"
if _FN_BF16SK:
    from vllm.model_executor.layers import fn_bf16sk as _fn_bf16sk

    _fn_bf16sk.register()
    init_logger(__name__).warning("FNBF16SK active: %d shapes", len(_fn_bf16sk.CONFIGS))


''' + anchor)
old = "        return self._gemm_impl(layer, x, layer.weight, bias)\n"
assert s.count(old) == 1
s = s.replace(old, '''        if (_FN_BF16SK and bias is None and x.dim() == 2
                and tuple(layer.weight.shape) in _fn_bf16sk.CONFIGS):
            return torch.ops.vllm.fn_bf16sk(x, layer.weight)
''' + old)
open(p, "w").write(s); print("patched")
