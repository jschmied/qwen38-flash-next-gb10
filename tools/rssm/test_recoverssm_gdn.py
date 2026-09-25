import sys, torch
sys.path.insert(0, "/opt/llm/runners/rssm")
from recoverssm_gdn import gdn_recoverssm_verify, GDNRecoverSSMCommitContext
from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import fused_sigmoid_gating_delta_rule_update
torch.manual_seed(0)
dev = "cuda"; H, HV, K, V, T = 16, 48, 128, 128, 4
nb = 10; N = 3; qlens = [4, 4, 3]
tot = sum(qlens)
qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(qlens), 0)), dtype=torch.int32, device=dev)
blocks = torch.tensor([2, 5, 7], dtype=torch.int32, device=dev)
ckpt = (torch.randn(nb, HV, V, K, device=dev) * 0.05).float()
q = torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16)
k = torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16)
v = torch.randn(1, tot, HV, V, device=dev, dtype=torch.bfloat16)
a = torch.randn(tot, HV, device=dev, dtype=torch.bfloat16)
b = torch.randn(tot, HV, device=dev, dtype=torch.bfloat16)
A_log = (torch.rand(HV, device=dev) * 2 - 1).float()
dt_bias = (torch.randn(HV, device=dev) * 0.5).bfloat16()
# native reference: every per-token state
ref_ckpt = ckpt.clone()
o_ref, states = fused_sigmoid_gating_delta_rule_update(
    A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v, initial_state=ref_ckpt, inplace_final_state=False,
    cu_seqlens=qsl.long(), ssm_state_indices=blocks.long(), use_qk_l2norm_in_kernel=True)
replay = torch.zeros(nb, HV, T, V + K + 1, device=dev)
work = ckpt.clone()
out = gdn_recoverssm_verify(A_log, a, b, dt_bias, q, k, v, checkpoint_state=work, replay_cache=replay,
                            query_start_loc=qsl, state_indices=blocks, spec_query_len=T)
o_ref = o_ref.reshape(out.shape)
err_o = ((out.float() - o_ref.float()).abs().max() / o_ref.float().abs().max()).item()
print(f"verify output rel max err vs native: {err_o:.2e}  (bf16 outputs)")
assert torch.equal(work, ckpt), "verify must not modify the checkpoint"
worst = 0.0
for n_acc in range(1, T + 1):
    w = ckpt.clone()
    conv = [torch.zeros(nb, 8, 3 + T - 1, device=dev, dtype=torch.bfloat16)]
    ctx = GDNRecoverSSMCommitContext.from_tensors(conv, [w], [replay], spec_query_len=T, max_num_reqs=8)
    acc = torch.tensor([min(n_acc, ql) for ql in qlens], dtype=torch.int32, device=dev)
    ctx.commit(acc, blocks, qsl)
    for i in range(N):
        na = int(acc[i]); tok = int(qsl[i]) + na - 1
        ref_state = states[tok]
        got = w[int(blocks[i])]
        e = ((got - ref_state).abs().max() / ref_state.abs().max()).item()
        worst = max(worst, e)
        untouched = [bi for bi in range(nb) if bi not in blocks.tolist()]
    assert torch.equal(w[untouched], ckpt[untouched]), "commit touched other blocks"
    print(f"accepted={n_acc}: committed-state rel max err vs native per-token state: {worst:.2e}")
print("PASS" if err_o < 2e-2 and worst < 1e-4 else "FAIL")
