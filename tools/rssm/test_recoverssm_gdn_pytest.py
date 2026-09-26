# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN RecoverSSM: verify against the native fused kernel, commit against the native per-token states, and the
boundary checks (malformed metadata must raise ValueError before any kernel reads or writes out of bounds)."""
import pytest
import torch

from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import (
    GDNRecoverSSMCommitContext,
    gdn_recoverssm_verify,
)
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
H, HV, K, V, T = 16, 48, 128, 128, 4
QLENS = (4, 4, 3)


def _inputs(nb: int = 10) -> dict:
    torch.manual_seed(0)
    dev = "cuda"
    tot = sum(QLENS)
    qsl = torch.tensor([0, 4, 8, 11], dtype=torch.int32, device=dev)
    return dict(
        A_log=(torch.rand(HV, device=dev) * 2 - 1).float(),
        a=torch.randn(tot, HV, device=dev, dtype=torch.bfloat16),
        b=torch.randn(tot, HV, device=dev, dtype=torch.bfloat16),
        dt_bias=(torch.randn(HV, device=dev) * 0.5).bfloat16(),
        q=torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16),
        k=torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16),
        v=torch.randn(1, tot, HV, V, device=dev, dtype=torch.bfloat16),
        checkpoint_state=(torch.randn(nb, HV, V, K, device=dev) * 0.05).float(),
        replay_cache=torch.zeros(nb, HV, T, V + K + 1, device=dev),
        query_start_loc=qsl,
        state_indices=torch.tensor([2, 5, 7], dtype=torch.int32, device=dev),
        spec_query_len=T,
    )


def _verify(inp: dict, **override):
    d = {**inp, **override}
    return gdn_recoverssm_verify(
        d["A_log"], d["a"], d["b"], d["dt_bias"], d["q"], d["k"], d["v"],
        checkpoint_state=d["checkpoint_state"], replay_cache=d["replay_cache"],
        query_start_loc=d["query_start_loc"], state_indices=d["state_indices"],
        spec_query_len=d["spec_query_len"], out=d.get("out"),
    )


def _native(inp: dict):
    return fused_sigmoid_gating_delta_rule_update(
        A_log=inp["A_log"], a=inp["a"], b=inp["b"], dt_bias=inp["dt_bias"], q=inp["q"], k=inp["k"], v=inp["v"],
        initial_state=inp["checkpoint_state"].clone(), inplace_final_state=False,
        cu_seqlens=inp["query_start_loc"].long(), ssm_state_indices=inp["state_indices"].long(),
        use_qk_l2norm_in_kernel=True,
    )


def _context(inp: dict, ckpt: torch.Tensor) -> GDNRecoverSSMCommitContext:
    nb = ckpt.shape[0]
    conv = [torch.zeros(nb, 8, 3 + T - 1, device="cuda", dtype=torch.bfloat16)]
    return GDNRecoverSSMCommitContext.from_tensors(
        conv, [ckpt], [inp["replay_cache"]], spec_query_len=T, max_num_reqs=8)


def test_verify_matches_native_and_keeps_the_checkpoint():
    inp = _inputs()
    before = inp["checkpoint_state"].clone()
    out = _verify(inp)
    ref, _ = _native(inp)
    torch.testing.assert_close(out.float(), ref.reshape(out.shape).float(), rtol=0, atol=0)
    assert torch.equal(inp["checkpoint_state"], before)


@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_commit_matches_native_per_token_state(accepted):
    inp = _inputs()
    _verify(inp)
    _, states = _native(inp)
    ckpt = inp["checkpoint_state"].clone()
    ctx = _context(inp, ckpt)
    acc = torch.tensor([min(accepted, q) for q in QLENS], dtype=torch.int32, device="cuda")
    ctx.commit(acc, inp["state_indices"], inp["query_start_loc"])
    for i, blk in enumerate(inp["state_indices"].tolist()):
        tok = int(inp["query_start_loc"][i]) + int(acc[i]) - 1
        torch.testing.assert_close(ckpt[blk], states[tok], rtol=1e-5, atol=1e-6)
    others = [b for b in range(ckpt.shape[0]) if b not in inp["state_indices"].tolist()]
    assert torch.equal(ckpt[others], inp["checkpoint_state"][others])


@pytest.mark.parametrize("case", ["short_qsl", "capacity", "hv_lt_h", "a_log", "out", "replay", "device"])
def test_verify_rejects_malformed_inputs(case):
    inp = _inputs()
    over, match = {
        "short_qsl": (dict(query_start_loc=inp["query_start_loc"][:-1]), "query metadata"),
        "capacity": (dict(state_indices=inp["state_indices"][:2], query_start_loc=inp["query_start_loc"][:3]),
                     "activation capacity"),
        "hv_lt_h": (dict(v=inp["v"][:, :, :8].contiguous(), a=inp["a"][:, :8].contiguous(),
                         b=inp["b"][:, :8].contiguous()), "positive multiple"),
        "a_log": (dict(A_log=torch.zeros(HV + 1, device="cuda")), "A_log"),
        "out": (dict(out=torch.empty(1, 11, HV, V + 1, device="cuda", dtype=torch.bfloat16)), "output shape"),
        "replay": (dict(replay_cache=torch.zeros(10, HV, T, V + K, device="cuda")), "replay buffer"),
        "device": (dict(A_log=inp["A_log"].cpu()), "same device"),
    }[case]
    with pytest.raises(ValueError, match=match):
        _verify(inp, **over)


def test_commit_rejects_partial_or_short_metadata():
    inp = _inputs()
    ctx = _context(inp, inp["checkpoint_state"].clone())
    acc = torch.ones(3, dtype=torch.int32, device="cuda")
    bt = torch.zeros(3, 4, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="align metadata is incomplete"):
        ctx.commit(acc, inp["state_indices"], inp["query_start_loc"], block_table=bt, mamba_block_size=16)
    with pytest.raises(ValueError, match="commit metadata"):
        ctx.commit(acc, inp["state_indices"], inp["query_start_loc"][:-1])
    with pytest.raises(ValueError, match="request mapping"):
        ctx.commit(acc, inp["state_indices"], inp["query_start_loc"],
                   request_indices=torch.zeros(2, dtype=torch.int32, device="cuda"))


def test_context_rejects_inconsistent_layers():
    inp = _inputs()
    ck = inp["checkpoint_state"]
    conv = torch.zeros(ck.shape[0], 8, 3 + T - 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at least one layer"):
        GDNRecoverSSMCommitContext.from_tensors([], [], [], spec_query_len=T, max_num_reqs=8)
    with pytest.raises(ValueError, match="differ"):
        GDNRecoverSSMCommitContext.from_tensors([conv], [ck, ck], [inp["replay_cache"]], spec_query_len=T,
                                                max_num_reqs=8)
    with pytest.raises(ValueError, match="block count"):
        GDNRecoverSSMCommitContext.from_tensors([conv[:5]], [ck], [inp["replay_cache"]], spec_query_len=T,
                                                max_num_reqs=8)
