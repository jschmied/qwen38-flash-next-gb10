#!/usr/bin/env python3
"""Sweep the fused QSA pre-indexer's tile and warp constants on synthetic prefill batches (GB10).

The wrapper hard-codes TILE_T_Q x TILE_H_Q = 2x2 (<=4096 tokens) / 2x4 and num_warps=1 on every device. This launches
`_qsa_pre_indexer_kernel` directly with the production geometry (4 index heads x 128, compress ratio 4, ring 8, mRoPE
[11,11,10] interleaved, rope tail in the ring) and times every (TILE_T_Q, TILE_H_Q, num_warps) combination, checking the
outputs (normalised q, compressed keys, ring) against the stock configuration. Usage: qsa_preindexer_tune.py (serving venv)"""
import statistics, itertools, torch, vllm  # noqa
from vllm.models.qwen4_exp.nvidia.ops import qsa_pre_indexer as P
from vllm.triton_utils import triton
DEV = "cuda"; HQ, D, CR, RING, PAGE = 4, 128, 4, 8, 400
torch.manual_seed(0)
def timeit(fn, n=10, reps=5):
    fn(); torch.cuda.synchronize(); out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True); s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); out.append(s.elapsed_time(e) * 1000 / n)
    return statistics.median(out)
def make(T_per_req, R):
    T = T_per_req * R
    q = (torch.randn(T, HQ * D, device=DEV) * 0.3).to(torch.bfloat16); k = (torch.randn(T, D, device=DEV) * 0.3).to(torch.bfloat16)
    pos1 = torch.cat([torch.arange(T_per_req, device=DEV) for _ in range(R)]); positions = pos1.unsqueeze(0).expand(3, T).contiguous()
    cos_sin = (torch.randn(T_per_req + 8, D // 2, device=DEV) * 0.7).to(torch.bfloat16).contiguous()
    qn = torch.ones(D, device=DEV, dtype=torch.bfloat16); kn = torch.ones(D, device=DEV, dtype=torch.bfloat16)
    qsl = torch.arange(0, T + 1, T_per_req, device=DEV, dtype=torch.int32); logical = pos1.to(torch.int32)
    ring_blocks = R + 1; state_cache = torch.zeros(ring_blocks, RING, 1, D + 12, device=DEV, dtype=torch.bfloat16)
    state_bt = torch.arange(1, R + 1, device=DEV, dtype=torch.int32).view(R, 1)
    state_slots = torch.full((T,), -1, device=DEV, dtype=torch.int32)
    for r in range(R):
        for p in range(max(0, T_per_req - RING), T_per_req): state_slots[r * T_per_req + p] = (r + 1) * RING + p % RING
    ngroups = T_per_req // CR; pages = triton.cdiv(ngroups, PAGE); comp_blocks = R * pages + 1
    comp_cache = torch.zeros(comp_blocks, PAGE, 1, D, device=DEV, dtype=torch.bfloat16)
    comp_slots = torch.full((T,), -1, device=DEV, dtype=torch.int32)
    for r in range(R):
        for g in range(ngroups):
            p = g * CR + CR - 1; blk = 1 + r * pages + g // PAGE; comp_slots[r * T_per_req + p] = blk * PAGE + g % PAGE
    work = torch.tensor([(r, i) for r in range(R) for i in range(max(ngroups, 1))], device=DEV, dtype=torch.int32).view(-1, 2)
    return dict(q=q, k=k, positions=positions, cos_sin=cos_sin, qn=qn, kn=kn, qsl=qsl, logical=logical, state_cache=state_cache,
                state_bt=state_bt, state_slots=state_slots, comp_cache=comp_cache, comp_slots=comp_slots, work=work, T=T)
def launch(m, tt, th, warps):
    q_out = torch.empty(m["T"], HQ, D, device=DEV, dtype=torch.bfloat16)
    sc = m["state_cache"].clone(); cc = m["comp_cache"].clone(); work = m["work"]; T = m["T"]
    num_q_work = triton.cdiv(T, tt) * triton.cdiv(HQ, th)
    def run():
        P._qsa_pre_indexer_kernel[(work.shape[0] + num_q_work,)](
            m["q"], m["q"].stride(0), m["k"], m["k"].stride(0), m["positions"], m["positions"].stride(0), m["positions"].stride(1),
            m["cos_sin"], m["qn"], m["kn"], 1e-6, q_out, q_out.stride(0), q_out.stride(1), sc, sc.stride(0), sc.stride(1),
            m["state_slots"], m["state_bt"], m["state_bt"].stride(0), m["qsl"], m["logical"], m["comp_slots"], work, cc, cc.stride(0), cc.stride(1),
            T, sc.shape[0], cc.shape[0], work.shape[0], HQ=HQ, D=D, TILE_T_Q=tt, TILE_H_Q=th, COMPRESS_RATIO=CR, STATE_SIZE=RING,
            COMP_PAGE_SIZE=PAGE, IS_2D_POSITIONS=True, IS_K_MROPE=True, CACHE_HAS_ROPE_POS=True, MROPE_H=11, MROPE_W=10, num_warps=warps)
    return run, q_out, sc, cc
for T_per_req, R in ((3813, 1), (2048, 2), (8192, 1), (545, 4)):
    m = make(T_per_req, R); stock = (2, 2, 1) if m["T"] <= 4096 else (2, 4, 1)
    run, q_ref, sc_ref, cc_ref = launch(m, *stock); t_stock = timeit(run)
    rows = []
    for tt, th, w in itertools.product((1, 2, 4, 8), (1, 2, 4), (1, 2, 4)):
        try:
            run, q_out, sc, cc = launch(m, tt, th, w); t = timeit(run)
            ok = torch.equal(q_out, q_ref) and torch.equal(sc, sc_ref) and torch.equal(cc, cc_ref)
            rows.append((t, tt, th, w, ok))
        except Exception as ex:
            rows.append((float("inf"), tt, th, w, str(ex).splitlines()[0][:50]))
    rows.sort(key=lambda r: r[0])
    print(f"pre-indexer T={m['T']} (R={R}): stock {stock} {t_stock:8.1f} us | best {rows[0][1:4]} {rows[0][0]:8.1f} us (x{t_stock/rows[0][0]:.2f}) identical={rows[0][4]}", flush=True)
    for t, tt, th, w, ok in rows[:5]: print(f"    TILE_T={tt} TILE_H={th} warps={w}: {t:8.1f} us identical={ok}", flush=True)
