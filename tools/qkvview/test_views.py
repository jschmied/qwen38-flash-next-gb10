"""A1 unit check: gdn_recoverssm_verify on strided views of a packed [T, qd+qd+vd(+z)] buffer, writing into out=,
must be bit-identical to the rearrange_mixed_qkv contiguous path, and must leave the replay record identical."""
import torch
from vllm.model_executor.layers.mamba.gdn.recoverssm_gdn import gdn_recoverssm_verify
torch.manual_seed(0); dev = "cuda"
H, HV, K, V, SQ, NB = 16, 48, 128, 128, 4, 6
for batch in (1, 2, 4):
    T = batch * SQ; qd = H * K; vd = HV * V; zd = vd
    packed = torch.randn(T, 2 * qd + vd + zd, device=dev, dtype=torch.bfloat16)
    mixed = packed[:, :2 * qd + vd]                       # non-contiguous rows, like the qkvz split
    a = torch.randn(T, HV, device=dev, dtype=torch.bfloat16); b = torch.randn(T, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=dev); dt = torch.randn(HV, device=dev)
    ck = torch.randn(NB, HV, V, K, device=dev) * 0.05
    qsl = torch.arange(0, T + 1, SQ, device=dev, dtype=torch.int32); si = torch.arange(1, batch + 1, device=dev, dtype=torch.int32)
    def ref():
        q, k, v = torch.split(mixed, [qd, qd, vd], dim=-1)
        fused = torch.cat([q.reshape(-1), k.reshape(-1), v.reshape(-1)])
        q = fused[:T * qd].view(1, T, -1, K); k = fused[T * qd:2 * T * qd].view(1, T, -1, K); v = fused[2 * T * qd:].view(1, T, -1, V)
        rep = torch.zeros(NB, HV, SQ, V + K + 1, device=dev)
        o = gdn_recoverssm_verify(A_log, a, b, dt, q, k, v, checkpoint_state=ck, replay_cache=rep, query_start_loc=qsl,
                                  state_indices=si, spec_query_len=SQ)
        return o.clone(), rep
    def new():
        q = mixed[:, :qd].view(1, T, -1, K); k = mixed[:, qd:2 * qd].view(1, T, -1, K); v = mixed[:, 2 * qd:2 * qd + vd].view(1, T, -1, V)
        rep = torch.zeros(NB, HV, SQ, V + K + 1, device=dev)
        core = torch.zeros(T + 3, HV, V, device=dev, dtype=torch.bfloat16)
        o = gdn_recoverssm_verify(A_log, a, b, dt, q, k, v, checkpoint_state=ck, replay_cache=rep, query_start_loc=qsl,
                                  state_indices=si, spec_query_len=SQ, out=core[:T].unsqueeze(0))
        assert o.data_ptr() == core.data_ptr()
        return core[:T].unsqueeze(0).clone(), rep
    (o1, r1), (o2, r2) = ref(), new()
    print(batch, "out equal", torch.equal(o1, o2), "replay equal", torch.equal(r1, r2))
    assert torch.equal(o1, o2) and torch.equal(r1, r2)
print("ALL EQUAL")
