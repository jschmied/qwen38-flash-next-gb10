"""main-tree port: let VocabParallelEmbedding.weight_loader load a BLOCK-scaled quantized head's
scale ([vocab/128, 1, hidden/128, 1], KFp8PbWo) — the generic loader asserts the vocab dim on it.
Needed for FP8_PB_WO lm_head checkpoints (Flash-Next fp8head: lm_head.weight_scale_inv [1940, 20]).
Target: vllm/model_executor/layers/vocab_parallel_embedding.py of the running interpreter. `off` removes."""
import os, sys
def _target():
    if os.environ.get("VLLM_VPE_PY"): return os.environ["VLLM_VPE_PY"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "model_executor/layers/vocab_parallel_embedding.py")
TARGET = _target()
ANCHOR = '''        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
'''
NEW = '''        # LMHEADSCALE (jschmied 2026-09-03): block scales of a block-quantized
        # head ([vocab/bn, 1, hidden/bk, 1], KFp8PbWo) shard along the vocab
        # dim in units of the block size, not the vocab.
        if param.data.ndim == 4 and loaded_weight.ndim == 4:
            rows = loaded_weight.shape[output_dim]
            block = self.org_vocab_size // rows
            assert block * rows == self.org_vocab_size, (
                f"block scale rows {rows} do not tile vocab {self.org_vocab_size}"
            )
            assert start_idx % block == 0 and shard_size % block == 0
            loaded_weight = loaded_weight.narrow(
                output_dim, start_idx // block, shard_size // block
            )
            param[: loaded_weight.shape[0]].data.copy_(loaded_weight)
            param[loaded_weight.shape[0] :].data.fill_(0)
            return

        # If param packed on the same dim we are sharding on, then
        # need to adjust offsets of loaded weight by pack_factor.
        if packed_dim is not None and packed_dim == output_dim:
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if NEW not in s: print("  lmheadscale not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  lmheadscale REMOVED")
else:
    if "LMHEADSCALE" in s: print("  lmheadscale already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  lmheadscale INSTALLED in", TARGET)
