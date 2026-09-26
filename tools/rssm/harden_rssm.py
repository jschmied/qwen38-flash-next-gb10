"""Bring recoverssm_gdn.py's boundary checks up to the KDA RecoverSSM implementation (review 2026-09-26):
ValueError instead of assert; the verify wrapper checks shapes, strides, HV >= H, query metadata, activation capacity
and devices; from_tensors checks layer lists, per-layer layout, block counts and devices before building raw pointer
tables; commit rejects partial align metadata, short query/request metadata and wrong devices.
Works on both copies (tools/rssm/ and tools/rssm/defer/). argv: <file> [<file> ...]. Idempotent (marker FNRSSMGUARD)."""
import sys

MARK = "FNRSSMGUARD"
HELPER = '''

def _require(cond: bool, msg: str) -> None:  # FNRSSMGUARD: boundary checks as in the KDA RecoverSSM ops
    if not cond:
        raise ValueError(f"GDN RecoverSSM: {msg}")
'''

VERIFY_BASE_OLD = '''    _, total, H, K = q.shape
    HV, V = v.shape[-2], v.shape[-1]
    a2 = a.reshape(total, HV); b2 = b.reshape(total, HV)
    nb = checkpoint_state.shape[0]
    assert checkpoint_state.shape[1:] == (HV, V, K), "GDN RecoverSSM checkpoint shape"
    assert replay_cache.shape == (nb, HV, spec_query_len, V + K + 1) and replay_cache.dtype == torch.float32
    assert HV % H == 0
    for t_ in (q, k):
        assert t_.stride(-1) == 1 and t_.stride(-2) == K
    assert v.stride(-1) == 1 and v.stride(-2) == V
'''
VERIFY_DEFER_OLD = '''    _, total, H, K = q.shape
    HV, V = v.shape[-2], v.shape[-1]
    a2 = a.reshape(total, HV); b2 = b.reshape(total, HV)
    nb = checkpoint_state.shape[0]
    assert checkpoint_state.shape[1:] == (HV, V, K), "GDN RecoverSSM checkpoint shape"
    deferred = pending is not None
    assert replay_cache.shape == (nb, HV, spec_query_len, replay_record_dim(K, V, deferred))
    assert replay_cache.dtype == torch.float32
    assert HV % H == 0
    for t_ in (q, k):
        assert t_.stride(-1) == 1 and t_.stride(-2) == K
    assert v.stride(-1) == 1 and v.stride(-2) == V
'''


def verify_new(defer: bool) -> str:
    rec = "replay_record_dim(K, V, deferred)" if defer else "V + K + 1"
    s = '''    _require(q.ndim == 4 and q.shape[0] == 1, "q must have shape [1, tokens, heads, dim]")  # FNRSSMGUARD
    _, total, H, K = q.shape
    _require(k.shape == q.shape, "q and k shapes differ")
    _require(v.ndim == 4 and tuple(v.shape[:2]) == (1, total), "v must have shape [1, tokens, value heads, dim]")
    HV, V = v.shape[-2], v.shape[-1]
    _require(H > 0 and HV >= H and HV % H == 0, "value heads must be a positive multiple of the q/k heads")
    _require(a.numel() == total * HV and b.numel() == total * HV, "gate or beta shape is incompatible")
    a2 = a.reshape(total, HV); b2 = b.reshape(total, HV)
    _require(a2.stride(1) == 1 and b2.stride(1) == 1, "gate and beta heads must be contiguous")
    _require(tuple(A_log.shape) == (HV,) and tuple(dt_bias.shape) == (HV,), "A_log or dt_bias shape is incompatible")
    _require(all(t_.stride(-1) == 1 and t_.stride(-2) == K for t_ in (q, k)), "q and k heads must be contiguous")
    _require(v.stride(-1) == 1 and v.stride(-2) == V, "v heads must be contiguous")
    _require(checkpoint_state.ndim == 4, "checkpoint must be four-dimensional")
    nb = checkpoint_state.shape[0]
    _require(tuple(checkpoint_state.shape[1:]) == (HV, V, K), "checkpoint shape is incompatible")
'''
    if defer:
        s += '''    deferred = pending is not None
'''
    s += f'''    _rec = (nb, HV, spec_query_len, {rec})
    _require(tuple(replay_cache.shape) == _rec, f"replay buffer needs shape {{_rec}}")
    _require(replay_cache.dtype == torch.float32, "replay buffer must use float32")
'''
    if defer:
        s += '''    _require(not deferred or (tuple(pending.shape) == (nb,) and pending.dtype == torch.int32),
             "pending counters need shape [blocks] and int32")
'''
    s += '''    _require(state_indices.ndim == 1, "state indices must be one-dimensional")
    _require(query_start_loc.ndim == 1 and query_start_loc.shape[0] == state_indices.shape[0] + 1,
             "query metadata is incompatible")
    _require(total <= state_indices.shape[0] * spec_query_len,
             "speculative decode input exceeds its activation capacity")
    _dev = q.device
    _require(all(t_.device == _dev for t_ in (k, v, a, b, A_log, dt_bias, checkpoint_state, replay_cache,
                                              query_start_loc, state_indices)
                 if t_ is not None)'''
    s += (''' and (out is None or out.device == _dev) and (pending is None or pending.device == _dev),
             "inputs must be on the same device")
''' if defer else ''' and (out is None or out.device == _dev),
             "inputs must be on the same device")
''')
    return s


OUT_OLD = '''    batch = state_indices.shape[0]
    if total == 0 or batch == 0:
        return out
'''
OUT_NEW = '''    _require(tuple(out.shape) == (1, total, HV, V) and out.stride()[2:] == (V, 1),
             "output shape or layout is incompatible")  # FNRSSMGUARD
    batch = state_indices.shape[0]
    if total == 0 or batch == 0:
        return out
'''


def from_old(defer: bool) -> str:
    rec_line = ('''            assert r.shape == (nb, HV, spec_query_len, replay_record_dim(K, V)) and r.dtype == torch.float32
''' if defer else '''            assert r.shape == (nb, HV, spec_query_len, V + K + 1) and r.dtype == torch.float32
''')
    return ('''        if not conv_dim_first:
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        nb, HV, V, K = ref.shape
        for s in checkpoints:
            assert s.shape == ref.shape and s.dtype == ref.dtype and s.stride()[1:] == ref.stride()[1:]
        for r in replays:
''' + rec_line + '''            assert r.stride()[1:] == replays[0].stride()[1:]
        conv_ref = conv_states[0]
        conv_dim, conv_len = conv_ref.shape[1:]
        hist = conv_len - spec_query_len + 1
        assert hist > 0, "GDN RecoverSSM conv state is shorter than its window"
''')


def from_new(defer: bool) -> str:
    rec = "replay_record_dim(K, V)" if defer else "V + K + 1"
    return f'''        _require(len(checkpoints) > 0, "commit requires at least one layer")  # FNRSSMGUARD
        _require(len(conv_states) == len(checkpoints) == len(replays), "conv, state and replay lists differ")
        if not conv_dim_first:
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        _require(ref.ndim == 4, "checkpoint must be four-dimensional")
        nb, HV, V, K = ref.shape
        _dev = ref.device
        for s in checkpoints:
            _require(s.shape == ref.shape and s.dtype == ref.dtype and s.stride()[1:] == ref.stride()[1:]
                     and s.device == _dev, "layers need matching checkpoints")
        _rec = (nb, HV, spec_query_len, {rec})
        for r in replays:
            _require(tuple(r.shape) == _rec and r.dtype == torch.float32 and r.stride()[1:] == replays[0].stride()[1:]
                     and r.device == _dev, f"layers need matching float32 replay buffers of shape {{_rec}}")
        conv_ref = conv_states[0]
        _require(conv_ref.ndim == 3 and conv_ref.shape[0] == nb,
                 "conv state must be [blocks, dim, window] with the checkpoint's block count")
        for c in conv_states:
            _require(c.shape == conv_ref.shape and c.dtype == conv_ref.dtype and c.stride()[1:] == conv_ref.stride()[1:]
                     and c.device == _dev, "layers need matching conv states")
        conv_dim, conv_len = conv_ref.shape[1:]
        hist = conv_len - spec_query_len + 1
        _require(hist > 0, "conv state is shorter than its window")
'''


COMMIT_OLD = '''        assert batch <= self.commit_lens.shape[0]
        if mamba_block_size is not None:
            assert mamba_block_size >= self.spec_query_len
'''
COMMIT_NEW = '''        _require(state_indices.ndim == 1, "state indices must be one-dimensional")  # FNRSSMGUARD
        _require(batch <= self.commit_lens.shape[0], "commit batch exceeds its plan capacity")
        _require(query_start_loc.ndim == 1 and query_start_loc.shape[0] == batch + 1, "commit metadata is incompatible")
        _require(request_indices is None or request_indices.shape[0] >= batch, "request mapping is too short")
        _require(num_accepted_tokens.ndim == 1
                 and (request_indices is not None or num_accepted_tokens.shape[0] >= batch),
                 "accepted-token counts are too short")
        _align = (block_table, num_computed_tokens, mamba_block_size)
        _require(all(x is None for x in _align) or all(x is not None for x in _align), "align metadata is incomplete")
        _require(mamba_block_size is None or mamba_block_size >= self.spec_query_len,
                 "align block size must cover one speculative window")
        _require(block_table is None or block_table.ndim == 2, "block table must be two-dimensional")
        _dev = self.checkpoints[0].device
        _require(all(t_ is None or t_.device == _dev for t_ in (num_accepted_tokens, state_indices, query_start_loc,
                                                                request_indices, block_table, num_computed_tokens)),
                 "commit inputs must be on the same device")
'''

for path in sys.argv[1:]:
    s = open(path).read()
    if MARK in s:
        print("already hardened:", path); continue
    defer = "def replay_record_dim" in s
    edits = [("from vllm.v1.attention.backends.utils import NULL_BLOCK_ID\n",
              "from vllm.v1.attention.backends.utils import NULL_BLOCK_ID\n" + HELPER),
             (VERIFY_DEFER_OLD if defer else VERIFY_BASE_OLD, verify_new(defer)),
             (OUT_OLD, OUT_NEW), (from_old(defer), from_new(defer)), (COMMIT_OLD, COMMIT_NEW)]
    for old, new in edits:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    assert "assert " not in s.split("def gdn_recoverssm_verify")[1].split("@dataclass")[0], "assert left in verify"
    open(path, "w").write(s); print("hardened:", path, "(deferred variant)" if defer else "")
