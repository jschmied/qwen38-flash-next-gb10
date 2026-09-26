# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RecoverSSM for Gated DeltaNet (Qwen-style GDN): speculative verify from one checkpoint without per-draft state
snapshots, and accepted-state recovery after sampling.

GDN port of the Kimi-K3 KDA RecoverSSM path (vllm/models/kimi_k3/nvidia/ops/recoverssm.py). Differences from KDA:
the gate is a scalar per value head (g = -exp(A_log) * softplus(a + dt_bias)), q/k heads are grouped over value
heads (HV // H), and q/k are L2-normalised inside the kernel when requested. The verify step caches, per token,
the delta-rule correction c_t = beta_t * (v_t - S'_t k_t) (V floats), the normalised key (K floats) and the
log-decay g_t (1 float) in one fp32 "replay" record. Recovery folds the accepted prefix:
    S_n = exp(sum_t g_t) S_0 + sum_t exp(sum_{s>t} g_s) c_t k_t^T
and writes the checkpoint once (plus the block-boundary state in align mode).

Deferred commit (FN_GDN_RECOVERSSM_DEFER=1): a row that crosses no block boundary is not committed after sampling;
its accepted count goes into a per-block ``pending`` counter, and the next step's verify replays those records
forward from the checkpoint, stores the result as the new checkpoint and then verifies the new drafts. The state
is then read and written once per step instead of read twice and written once. Rows that cross a block boundary
are committed immediately, so every block-boundary (prefix-cacheable) state is exact. Each value slice keeps its
own copy of k and g in the record, because the next verify rewrites the record while sibling slices still read it.
"""
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

DEFERRED = os.environ.get("FN_GDN_RECOVERSSM_DEFER", "") == "1"
_BV = 32  # value tile of the verify and commit kernels; the deferred record keeps k/g once per value tile


def replay_record_dim(K: int, V: int, deferred: bool | None = None) -> int:
    """Floats per token in the replay record: correction (V), then k (K) and g (1) once, or once per value tile."""
    deferred = DEFERRED if deferred is None else deferred
    return V + (triton.cdiv(V, _BV) if deferred else 1) * (K + 1)


@triton.jit
def _gdn_recoverssm_verify_kernel(
    q_ptr, k_ptr, v_ptr, a_ptr, b_ptr, A_log_ptr, dt_bias_ptr,
    state_ptr, replay_ptr, out_ptr, query_start_loc_ptr, state_indices_ptr,
    scale, softplus_beta, softplus_threshold, null_block_id,
    stride_q_token, stride_k_token, stride_v_token, stride_a_token, stride_b_token,
    stride_state_block, stride_state_head, stride_state_v, stride_state_k,
    stride_replay_block, stride_replay_head, stride_replay_pos, stride_replay_dim,
    stride_out_token, stride_qsl, stride_si, pending_ptr,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, SPEC_QUERY_LEN: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
    DEFERRED: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_hv = tl.program_id(2)
    pid_h = pid_hv // (HV // H)
    bos = tl.load(query_start_loc_ptr + pid_b * stride_qsl).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + (pid_b + 1) * stride_qsl).to(tl.int64)
    query_len = eos - bos
    state_idx = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    if state_idx <= null_block_id:
        for t in tl.static_range(SPEC_QUERY_LEN):
            tl.store(out_ptr + (bos + t) * stride_out_token + pid_hv * V + offs_v,
                     tl.zeros([BV], dtype=tl.float32), mask=(t < query_len) & mask_v)
        return
    st_ptrs = (state_ptr + state_idx * stride_state_block + pid_hv * stride_state_head
               + offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k)
    state = tl.load(st_ptrs, mask=mask_state, other=0.0).to(tl.float32)
    neg_A = -tl.exp(tl.load(A_log_ptr + pid_hv).to(tl.float32))
    dt_bias = tl.load(dt_bias_ptr + pid_hv).to(tl.float32)
    rec_base = replay_ptr + state_idx * stride_replay_block + pid_hv * stride_replay_head
    if DEFERRED:
        kg_off = V + pid_v * (K + 1)
        # Fold the previous step's accepted tokens (forward) and store them as the new checkpoint.
        n_pend = tl.load(pending_ptr + state_idx)
        for r in range(n_pend):
            prev = rec_base + r * stride_replay_pos
            c0 = tl.load(prev + offs_v * stride_replay_dim, mask=mask_v, other=0.0)
            k0 = tl.load(prev + (kg_off + offs_k) * stride_replay_dim, mask=mask_k, other=0.0)
            g0 = tl.load(prev + (kg_off + K) * stride_replay_dim)
            state = state * tl.exp(g0) + c0[:, None] * k0[None, :]
        if n_pend > 0:
            tl.store(st_ptrs, state.to(state_ptr.dtype.element_ty), mask=mask_state)
        tl.debug_barrier()  # every read of the previous records before this step rewrites them
    else:
        kg_off = V
    for t in tl.static_range(SPEC_QUERY_LEN):
        valid = t < query_len
        tok = bos + t
        q = tl.load(q_ptr + tok * stride_q_token + pid_h * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + tok * stride_k_token + pid_h * K + offs_k, mask=valid & mask_k, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + tok * stride_v_token + pid_hv * V + offs_v, mask=valid & mask_v, other=0.0).to(tl.float32)
        x = tl.load(a_ptr + tok * stride_a_token + pid_hv, mask=valid, other=0.0).to(tl.float32) + dt_bias
        sp = tl.where(softplus_beta * x <= softplus_threshold,
                      (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x)), x)
        g = neg_A * sp
        beta = tl.sigmoid(tl.load(b_ptr + tok * stride_b_token + pid_hv, mask=valid, other=0.0).to(tl.float32))
        if USE_QK_L2NORM:
            q = q * tl.rsqrt(tl.sum(q * q) + 1e-6)
            k = k * tl.rsqrt(tl.sum(k * k) + 1e-6)
        q = q * scale
        new_state = state * tl.exp(g)
        corr = (v - tl.sum(new_state * k[None, :], 1)) * beta
        new_state = new_state + corr[:, None] * k[None, :]
        state = tl.where(valid, new_state, state)
        o = tl.sum(state * q[None, :], 1)
        tl.store(out_ptr + tok * stride_out_token + pid_hv * V + offs_v, o.to(out_ptr.dtype.element_ty),
                 mask=valid & mask_v)
        rec = rec_base + t * stride_replay_pos
        tl.store(rec + offs_v * stride_replay_dim, corr, mask=valid & mask_v)
        if DEFERRED:
            tl.store(rec + (kg_off + offs_k) * stride_replay_dim, k, mask=valid & mask_k)
            tl.store(rec + (kg_off + K) * stride_replay_dim, g, mask=valid)
        elif pid_v == 0:
            tl.store(rec + (V + offs_k) * stride_replay_dim, k, mask=valid & mask_k)
            tl.store(rec + (V + K) * stride_replay_dim, g, mask=valid)


def gdn_recoverssm_verify(
    A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
    checkpoint_state: torch.Tensor, replay_cache: torch.Tensor,
    query_start_loc: torch.Tensor, state_indices: torch.Tensor, spec_query_len: int,
    scale: float | None = None, softplus_beta: float = 1.0, softplus_threshold: float = 20.0,
    use_qk_l2norm_in_kernel: bool = True, out: torch.Tensor | None = None,
    pending: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify a GDN speculative window against its checkpoint without modifying the checkpoint.
    q, k: [1, T, H, K]; v: [1, T, HV, V]; a, b: [T, HV] (or [1, T, HV]); checkpoint_state: [blocks, HV, V, K];
    replay_cache: [blocks, HV, spec_query_len, replay_record_dim(K, V)] fp32; pending: [blocks] int32 accepted
    counts not yet folded into the checkpoint (deferred commit), or None. Returns out [1, T, HV, V]."""
    _, total, H, K = q.shape
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
    if scale is None:
        scale = K ** -0.5
    if out is None:
        out = torch.empty(1, total, HV, V, dtype=v.dtype, device=v.device)
    batch = state_indices.shape[0]
    if total == 0 or batch == 0:
        return out
    BK = triton.next_power_of_2(K); BV = _BV
    grid = (triton.cdiv(V, BV), batch, HV)
    _gdn_recoverssm_verify_kernel[grid](
        q, k, v, a2, b2, A_log, dt_bias, checkpoint_state, replay_cache, out, query_start_loc, state_indices,
        scale, softplus_beta, softplus_threshold, NULL_BLOCK_ID,
        q.stride(1), k.stride(1), v.stride(1), a2.stride(0), b2.stride(0),
        checkpoint_state.stride(0), checkpoint_state.stride(1), checkpoint_state.stride(2), checkpoint_state.stride(3),
        replay_cache.stride(0), replay_cache.stride(1), replay_cache.stride(2), replay_cache.stride(3),
        out.stride(1), query_start_loc.stride(0), state_indices.stride(0),
        pending if deferred else state_indices,
        H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, SPEC_QUERY_LEN=spec_query_len,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel, DEFERRED=deferred, num_warps=4, num_stages=2,
    )
    return out


@triton.jit
def _commit_gdn_state_kernel(
    state_ref_ptr, state_base_addrs_ptr, state_block_strides_ptr,
    replay_ref_ptr, replay_base_addrs_ptr, replay_block_strides_ptr,
    state_indices_ptr, commit_lens_ptr, final_state_indices_ptr, boundary_state_indices_ptr,
    boundary_recovery_lens_ptr, null_block_id,
    stride_state_head, stride_state_v, stride_state_k,
    stride_replay_head, stride_replay_pos, stride_replay_dim, stride_si,
    K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    NUM_HEADS: tl.constexpr, ALIGN_MODE: tl.constexpr, DEFERRED: tl.constexpr,
):
    pid_v = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_lh = tl.program_id(2)
    pid_l = pid_lh // NUM_HEADS
    pid_h = pid_lh % NUM_HEADS
    src = tl.load(state_indices_ptr + pid_b * stride_si).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + pid_b)
    if n == 0:
        return
    final_idx = tl.load(final_state_indices_ptr + pid_b).to(tl.int64)
    boundary_idx = tl.load(boundary_state_indices_ptr + pid_b).to(tl.int64)
    boundary_len = tl.load(boundary_recovery_lens_ptr + pid_b)
    if final_idx <= null_block_id:
        return
    if DEFERRED:
        if (boundary_idx <= null_block_id) & (final_idx == src):
            return  # deferred: the next verify folds these records
        kg_off = V + pid_v * (K + 1)
    else:
        kg_off = V
    state_ptr = tl.load(state_base_addrs_ptr + pid_l).to(tl.pointer_type(state_ref_ptr.dtype.element_ty))
    sbs = tl.load(state_block_strides_ptr + pid_l)
    replay_ptr = tl.load(replay_base_addrs_ptr + pid_l).to(tl.pointer_type(replay_ref_ptr.dtype.element_ty))
    rbs = tl.load(replay_block_strides_ptr + pid_l)
    rec_base = replay_ptr + src * rbs + pid_h * stride_replay_head
    offs_k = tl.arange(0, BK)
    offs_v = pid_v * BV + tl.arange(0, BV)
    mask_k = offs_k < K
    mask_v = offs_v < V
    mask_state = mask_v[:, None] & mask_k[None, :]
    sp = (offs_v[:, None] * stride_state_v + offs_k[None, :] * stride_state_k)
    s0 = tl.load(state_ptr + src * sbs + pid_h * stride_state_head + sp, mask=mask_state, other=0.0).to(tl.float32)
    fin_decay = 1.0
    fin_corr = tl.zeros([BV, BK], tl.float32)
    bnd_decay = 1.0
    bnd_corr = tl.zeros([BV, BK], tl.float32)
    for r in range(n):
        t = n - r - 1
        rec = rec_base + t * stride_replay_pos
        c = tl.load(rec + offs_v * stride_replay_dim, mask=mask_v, other=0.0)
        kk = tl.load(rec + (kg_off + offs_k) * stride_replay_dim, mask=mask_k, other=0.0)
        g = tl.load(rec + (kg_off + K) * stride_replay_dim)
        upd = c[:, None] * kk[None, :]
        dec = tl.exp(g)
        fin_corr += upd * fin_decay
        fin_decay *= dec
        if ALIGN_MODE:
            before = t < boundary_len
            bnd_corr += tl.where(before, upd * bnd_decay, 0.0)
            bnd_decay *= tl.where(before, dec, 1.0)
    if ALIGN_MODE:
        bs = s0 * bnd_decay + bnd_corr
        tl.store(state_ptr + boundary_idx * sbs + pid_h * stride_state_head + sp,
                 bs.to(state_ref_ptr.dtype.element_ty), mask=mask_state & (boundary_idx > null_block_id))
    fs = s0 * fin_decay + fin_corr
    tl.store(state_ptr + final_idx * sbs + pid_h * stride_state_head + sp,
             fs.to(state_ref_ptr.dtype.element_ty), mask=mask_state)


@triton.jit
def _set_pending_kernel(pending_ptr, state_indices_ptr, commit_lens_ptr, final_state_indices_ptr,
                        boundary_state_indices_ptr, null_block_id, stride_si):
    """Deferred rows record their accepted count; committed rows (boundary crossed) clear source and target."""
    b = tl.program_id(0)
    src = tl.load(state_indices_ptr + b * stride_si).to(tl.int64)
    if src <= null_block_id:
        return
    n = tl.load(commit_lens_ptr + b)
    fin = tl.load(final_state_indices_ptr + b).to(tl.int64)
    bnd = tl.load(boundary_state_indices_ptr + b).to(tl.int64)
    if (bnd <= null_block_id) & (fin == src):
        tl.store(pending_ptr + src, n)
    else:
        tl.store(pending_ptr + src, 0)
        if fin > null_block_id:
            tl.store(pending_ptr + fin, 0)


@dataclass
class GDNRecoverSSMCommitContext:
    conv_states: tuple[torch.Tensor, ...]
    conv_state_base_addrs: torch.Tensor
    conv_state_block_strides: torch.Tensor
    conv_state_dim_strides: torch.Tensor
    conv_state_token_strides: torch.Tensor
    conv_history_len: int
    checkpoints: tuple[torch.Tensor, ...]
    state_base_addrs: torch.Tensor
    state_block_strides: torch.Tensor
    replays: tuple[torch.Tensor, ...]
    replay_base_addrs: torch.Tensor
    replay_block_strides: torch.Tensor
    commit_lens: torch.Tensor
    final_state_indices: torch.Tensor
    boundary_state_indices: torch.Tensor
    boundary_recovery_lens: torch.Tensor
    spec_query_len: int
    pending: torch.Tensor | None = None  # [blocks] int32, deferred commit only

    @classmethod
    def from_tensors(cls, conv_states: Sequence[torch.Tensor], checkpoints: Sequence[torch.Tensor],
                     replays: Sequence[torch.Tensor], *, spec_query_len: int, max_num_reqs: int,
                     conv_dim_first: bool = True) -> "GDNRecoverSSMCommitContext":
        if not conv_dim_first:
            conv_states = [s.transpose(-1, -2) for s in conv_states]
        ref = checkpoints[0]
        nb, HV, V, K = ref.shape
        for s in checkpoints:
            assert s.shape == ref.shape and s.dtype == ref.dtype and s.stride()[1:] == ref.stride()[1:]
        for r in replays:
            assert r.shape == (nb, HV, spec_query_len, replay_record_dim(K, V)) and r.dtype == torch.float32
            assert r.stride()[1:] == replays[0].stride()[1:]
        conv_ref = conv_states[0]
        conv_dim, conv_len = conv_ref.shape[1:]
        hist = conv_len - spec_query_len + 1
        assert hist > 0, "GDN RecoverSSM conv state is shorter than its window"
        dev = ref.device
        addr = lambda ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        bstr = lambda ts: torch.tensor([t.stride(0) for t in ts], dtype=torch.int64, device=dev)
        z = lambda: torch.empty(max_num_reqs, dtype=torch.int32, device=dev)
        return cls(tuple(conv_states), addr(conv_states), bstr(conv_states),
                   torch.tensor([s.stride(1) for s in conv_states], dtype=torch.int64, device=dev),
                   torch.tensor([s.stride(2) for s in conv_states], dtype=torch.int64, device=dev),
                   hist, tuple(checkpoints), addr(checkpoints), bstr(checkpoints),
                   tuple(replays), addr(replays), bstr(replays), z(), z(), z(), z(), spec_query_len,
                   torch.zeros(nb, dtype=torch.int32, device=dev) if DEFERRED else None)

    @classmethod
    def create(cls, layers: Sequence[Any], *, spec_query_len: int, max_num_reqs: int) -> "GDNRecoverSSMCommitContext":
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        if any(len(layer.kv_cache) != 3 for layer in layers):
            raise ValueError("GDN RecoverSSM pages must contain conv, state and replay")
        return cls.from_tensors([l.kv_cache[0] for l in layers], [l.kv_cache[1] for l in layers],
                                [l.kv_cache[2] for l in layers], spec_query_len=spec_query_len,
                                max_num_reqs=max_num_reqs, conv_dim_first=is_conv_state_dim_first())

    def clear_pending(self, state_indices: torch.Tensor) -> None:
        """Forget pending records of blocks that start a prefill: a reused block must not replay a previous
        owner's records. Called before the step's forward, for the prefill rows only."""
        if self.pending is not None and state_indices.numel():
            self.pending.index_fill_(0, state_indices.reshape(-1).long().clamp_min(0), 0)

    def commit(self, num_accepted_tokens: torch.Tensor, state_indices: torch.Tensor,
               query_start_loc: torch.Tensor, request_indices: torch.Tensor | None = None,
               block_table: torch.Tensor | None = None, num_computed_tokens: torch.Tensor | None = None,
               mamba_block_size: int | None = None, commit_conv: bool = True) -> None:
        """Fold accepted GDN and convolution inputs into every layer's checkpoint."""
        from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
            _compact_conv_state_kernel, _prepare_commit_plan_kernel)
        batch = state_indices.shape[0]
        if batch == 0:
            return
        assert batch <= self.commit_lens.shape[0]
        if mamba_block_size is not None:
            assert mamba_block_size >= self.spec_query_len
        bt_stride = (0, 0) if block_table is None else block_table.stride()
        _prepare_commit_plan_kernel[(batch,)](
            num_accepted_tokens, request_indices, state_indices, query_start_loc, block_table, num_computed_tokens,
            self.commit_lens, self.final_state_indices, self.boundary_state_indices, self.boundary_recovery_lens,
            NULL_BLOCK_ID, mamba_block_size or 1, block_table.shape[1] if block_table is not None else 1,
            num_accepted_tokens.stride(0), request_indices.stride(0) if request_indices is not None else 0,
            state_indices.stride(0), query_start_loc.stride(0), bt_stride[0], bt_stride[1],
            0 if num_computed_tokens is None else num_computed_tokens.stride(0),
            SPEC_QUERY_LEN=self.spec_query_len, num_warps=1)
        num_layers = len(self.checkpoints)
        if commit_conv:
            conv_ref = self.conv_states[0]
            conv_dim = conv_ref.shape[1]
            _compact_conv_state_kernel[(triton.cdiv(conv_dim, 256), batch, num_layers)](
                conv_ref, self.conv_state_base_addrs, self.conv_state_block_strides, self.conv_state_dim_strides,
                self.conv_state_token_strides, state_indices, self.commit_lens, self.final_state_indices,
                self.boundary_state_indices, self.boundary_recovery_lens, NULL_BLOCK_ID, conv_dim,
                self.conv_history_len, state_indices.stride(0), BLOCK_D=256,
                BLOCK_HISTORY=triton.next_power_of_2(self.conv_history_len),
                ALIGN_MODE=block_table is not None, num_warps=4)
        ref = self.checkpoints[0]
        _, HV, V, K = ref.shape
        BK = triton.next_power_of_2(K); BV = _BV
        rr = self.replays[0]
        if self.pending is not None:
            _set_pending_kernel[(batch,)](
                self.pending, state_indices, self.commit_lens, self.final_state_indices,
                self.boundary_state_indices, NULL_BLOCK_ID, state_indices.stride(0), num_warps=1)
        _commit_gdn_state_kernel[(triton.cdiv(V, BV), batch, num_layers * HV)](
            ref, self.state_base_addrs, self.state_block_strides, rr, self.replay_base_addrs,
            self.replay_block_strides, state_indices, self.commit_lens, self.final_state_indices,
            self.boundary_state_indices, self.boundary_recovery_lens, NULL_BLOCK_ID,
            ref.stride(1), ref.stride(2), ref.stride(3), rr.stride(1), rr.stride(2), rr.stride(3),
            state_indices.stride(0), K=K, V=V, BK=BK, BV=BV, NUM_HEADS=HV,
            ALIGN_MODE=block_table is not None, DEFERRED=self.pending is not None, num_warps=4, num_stages=2)


__all__ = ["DEFERRED", "GDNRecoverSSMCommitContext", "gdn_recoverssm_verify", "replay_record_dim"]
