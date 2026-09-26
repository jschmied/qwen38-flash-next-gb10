"""Deferred vs immediate RecoverSSM commit for GDN, several steps, mode none and align (small blocks, boundary
crossings). Both paths run the same kernels file, loaded twice with FN_GDN_RECOVERSSM_DEFER off and on.
Checks per step: verify outputs; after the run: every block-boundary state and the final running state."""
import importlib.util, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(deferred):
    os.environ["FN_GDN_RECOVERSSM_DEFER"] = "1" if deferred else ""
    spec = importlib.util.spec_from_file_location(f"rssm_{int(deferred)}", os.path.join(HERE, "recoverssm_gdn.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.DEFERRED == deferred
    return m


IMM, DEF = load(False), load(True)
dev = "cuda"; H, HV, K, V, T = 16, 48, 128, 128, 4
NB = 40; NREQ = 3; STEPS = 7


def run(mod, align, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda *s, dt=torch.float32: torch.randn(*s, device=dev, generator=g).to(dt)
    A_log = (torch.rand(HV, device=dev, generator=g) * 2 - 1).float()
    dt_bias = (rnd(HV) * 0.5).bfloat16()
    ckpt = rnd(NB, HV, V, K) * 0.05
    rec_dim = mod.replay_record_dim(K, V)
    replay = torch.zeros(NB, HV, T, rec_dim, device=dev)
    conv = [torch.zeros(NB, 8, 3 + T - 1, device=dev, dtype=torch.bfloat16)]
    ctx = mod.GDNRecoverSSMCommitContext.from_tensors(conv, [ckpt], [replay], spec_query_len=T, max_num_reqs=8)
    bs = 6                                            # mamba block size (align): crossings every few steps
    bt = torch.arange(1, 1 + NREQ * 12, device=dev, dtype=torch.int32).reshape(NREQ, 12)
    ncomp = torch.tensor([0, 3, 5], device=dev, dtype=torch.int32)
    acc_g = torch.Generator().manual_seed(seed + 1)
    outs = []
    for step in range(STEPS):
        qlens = [T] * NREQ
        tot = sum(qlens)
        qsl = torch.tensor([0] + [T * (i + 1) for i in range(NREQ)], dtype=torch.int32, device=dev)
        q = rnd(1, tot, H, K, dt=torch.bfloat16); k = rnd(1, tot, H, K, dt=torch.bfloat16)
        v = rnd(1, tot, HV, V, dt=torch.bfloat16); a = rnd(tot, HV, dt=torch.bfloat16); b = rnd(tot, HV, dt=torch.bfloat16)
        if align:
            src = bt[torch.arange(NREQ, device=dev), (ncomp // bs).long()].contiguous()
        else:
            src = bt[:, 0].contiguous()
        o = mod.gdn_recoverssm_verify(A_log, a, b, dt_bias, q, k, v, checkpoint_state=ckpt, replay_cache=replay,
                                      query_start_loc=qsl, state_indices=src, spec_query_len=T, pending=ctx.pending)
        outs.append(o.float().clone())
        acc = torch.randint(1, T + 1, (NREQ,), generator=acc_g).to(torch.int32).to(dev)
        if align:
            ctx.commit(acc, src, qsl, block_table=bt, num_computed_tokens=ncomp, mamba_block_size=bs)
            ncomp = ncomp + acc
        else:
            ctx.commit(acc, src, qsl)
    # flush: a one-token verify folds the pending records into the checkpoint (its own token is not committed)
    if ctx.pending is not None:
        src = (bt[torch.arange(NREQ, device=dev), (ncomp // bs).long()] if align else bt[:, 0]).contiguous()
        qsl1 = torch.arange(NREQ + 1, dtype=torch.int32, device=dev)
        mod.gdn_recoverssm_verify(A_log, rnd(NREQ, HV, dt=torch.bfloat16), rnd(NREQ, HV, dt=torch.bfloat16), dt_bias,
                                  rnd(1, NREQ, H, K, dt=torch.bfloat16), rnd(1, NREQ, H, K, dt=torch.bfloat16),
                                  rnd(1, NREQ, HV, V, dt=torch.bfloat16), checkpoint_state=ckpt, replay_cache=replay,
                                  query_start_loc=qsl1, state_indices=src, spec_query_len=T, pending=ctx.pending)
    return outs, ckpt, (ncomp if align else None)


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


ok = True
for align in (False, True):
    oi, ci, nc = run(IMM, align, 7)
    od, cd, _ = run(DEF, align, 7)
    eo = max(rel(x, y) for x, y in zip(od, oi))
    touched = sorted(set(range(1, 1 + NREQ * 12)))
    es = rel(cd[touched], ci[touched])
    print(f"align={align}: max verify-output rel diff {eo:.2e} over {len(oi)} steps; "
          f"all block states rel diff {es:.2e}" + (f"; tokens per request {nc.tolist()}" if align else ""))
    ok &= eo < 2e-2 and es < 1e-5
# stale pending must be cleared for a reused block
DEFm = DEF
nb = 4
ck = torch.randn(nb, HV, V, K, device=dev) * 0.05
rp = torch.randn(nb, HV, T, DEFm.replay_record_dim(K, V), device=dev)
cx = DEFm.GDNRecoverSSMCommitContext.from_tensors([torch.zeros(nb, 8, 3 + T - 1, device=dev, dtype=torch.bfloat16)],
                                                   [ck], [rp], spec_query_len=T, max_num_reqs=4)
cx.pending[2] = 3
cx.clear_pending(torch.tensor([2, -1], device=dev, dtype=torch.int32))
ok &= int(cx.pending[2]) == 0
print("stale pending cleared:", int(cx.pending[2]) == 0)
print("PASS" if ok else "FAIL")
