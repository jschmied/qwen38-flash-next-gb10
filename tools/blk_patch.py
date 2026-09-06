# Small-block overlay for hybrid Mamba/attention prefix caching. Usage: blk_patch.py apply|revert <venv>
# FN_KEEP_BLOCK=1: keep the user's --block-size instead of raising it to the Mamba page; pad every attention page
# (row-aligned) up to the Mamba page so all groups share one page size. Cost: padded attention pages; gain: prefix-cache
# hits and Mamba checkpoints every block_size tokens instead of every ~1,600.
import sys, shutil, os
mode, venv = sys.argv[1], sys.argv[2]; sp = f"{venv}/lib/python3.12/site-packages/vllm"
PLAT = f"{sp}/platforms/interface.py"; KVU = f"{sp}/v1/core/kv_cache_utils.py"; FILES = [PLAT, KVU]
def rep(s, old, new):
    assert s.count(old) == 1, (old[:80], s.count(old)); return s.replace(old, new)
A1 = '''        if cache_config.block_size < attn_block_size:
            cache_config.block_size = attn_block_size
            logger.info(
                "Setting attention block size to %d tokens "
                "to ensure that attention page size is >= mamba page size.",
                attn_block_size,
            )
'''
B1 = '''        if (
            os.environ.get("FN_KEEP_BLOCK") == "1"
            and cache_config.block_size < attn_block_size
        ):
            # FN_KEEP_BLOCK: keep the user's block size; attention pages are padded up
            # to the Mamba page later in unify_kv_cache_spec_page_size (row-aligned).
            logger.info(
                "FN_KEEP_BLOCK: keeping attention block size %d (derived minimum "
                "was %d); attention pages will be padded to the mamba page.",
                cache_config.block_size,
                attn_block_size,
            )
            if cache_config.mamba_cache_mode == "align":
                cache_config.mamba_block_size = cache_config.block_size
            return
''' + A1
A2 = '''    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        # All layers have the same page size, no need to unify.
        return kv_cache_spec

    max_page_size = max(page_sizes)
'''
B2 = A2 + '''    if os.environ.get("FN_KEEP_BLOCK") == "1":
        # FN_KEEP_BLOCK: pad every page (Mamba, MLA, ring) to one shared page that is a
        # multiple of every attention row so MLA's flat row view stays valid, and a
        # multiple of every smaller attention page so those can scale by block instead.
        import math

        rows = []
        for _spec in kv_cache_spec.values():
            if isinstance(_spec, AttentionSpec):
                _row = _spec.page_size_bytes // _spec.block_size
                rows.append(_row)
                if _spec.page_size_bytes < max_page_size:
                    rows.append(_spec.page_size_bytes)
        _lcm = 1
        for _r in rows:
            _lcm = math.lcm(_lcm, max(_r, 1))
        shared = -(-max_page_size // _lcm) * _lcm
        logger.info(
            "FN_KEEP_BLOCK: shared page %d bytes (max page %d, row lcm %d) for %d layers",
            shared, max_page_size, _lcm, len(kv_cache_spec),
        )
        out: dict[str, KVCacheSpec] = {}
        for _name, _spec in kv_cache_spec.items():
            if _spec.page_size_bytes == shared:
                out[_name] = _spec
            elif isinstance(_spec, MambaSpec) or not isinstance(_spec, AttentionSpec):
                out[_name] = replace(_spec, page_size_padded=shared)
            elif shared % _spec.page_size_bytes == 0 and not isinstance(_spec, MLAAttentionSpec):
                out[_name] = replace(_spec, block_size=_spec.block_size * (shared // _spec.page_size_bytes))
            else:
                out[_name] = replace(_spec, page_size_padded=shared)
            assert out[_name].page_size_bytes == shared, (_name, out[_name].page_size_bytes, shared)
        return out
'''
if mode == "apply":
    for p in FILES: assert not os.path.exists(p + ".orig-blk"), p
    for p in FILES: shutil.copy2(p, p + ".orig-blk")
    s = open(PLAT).read(); assert "FN_KEEP_BLOCK" not in s; s = rep(s, A1, B1); open(PLAT, "w").write(s)
    s = open(KVU).read(); assert "FN_KEEP_BLOCK" not in s; s = rep(s, A2, B2)
    assert "\nimport os\n" in s and "AttentionSpec" in s and "MLAAttentionSpec" in s and "MambaSpec" in s and "replace(" in s
    open(KVU, "w").write(s); print("applied")
else:
    for p in FILES:
        b = p + ".orig-blk"; assert os.path.exists(b), b; shutil.copy2(b, p); os.remove(b)
    print("reverted")
