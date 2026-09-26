"""Standalone bench of _gdn_recoverssm_verify_kernel launch configs (agenda 6). L2 flushed before each call,
CUDA-graph replay timing, c=1 (batch 1) and c=4 (batch 4), T=4, H=16, HV=48, K=V=128.
Every config's out + replay record must be bit-identical to the shipped BV=32/num_warps=4 config."""
import itertools, json, sys
import torch
sys.path.insert(0, "/opt/llm/runners/rssm")
import recoverssm_gdn as R
from vllm.triton_utils import triton

torch.manual_seed(0)
dev = "cuda"
H, HV, K, V, T = 16, 48, 128, 128, 4
NB = 64
scratch = torch.empty(64 << 20, dtype=torch.uint8, device=dev)


def inputs(batch):
    tot = batch * T
    q = torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, tot, H, K, device=dev, dtype=torch.bfloat16)
    v = torch.randn(1, tot, HV, V, device=dev, dtype=torch.bfloat16)
    a = torch.randn(tot, HV, device=dev, dtype=torch.bfloat16)
    b = torch.randn(tot, HV, device=dev, dtype=torch.bfloat16)
    A_log = torch.randn(HV, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(HV, device=dev, dtype=torch.float32)
    st = torch.randn(NB, HV, V, K, device=dev, dtype=torch.float32) * 0.1
    rep = torch.zeros(NB, HV, T, V + K + 1, device=dev, dtype=torch.float32)
    qsl = torch.arange(0, tot + 1, T, device=dev, dtype=torch.int32)
    si = torch.arange(1, batch + 1, device=dev, dtype=torch.int32) * 3
    return q, k, v, a, b, A_log, dt_bias, st, rep, qsl, si


def launch(args, BV, nw, out, rep):
    q, k, v, a2, b2, A_log, dt_bias, st, _, qsl, si = args
    batch = si.shape[0]
    R._gdn_recoverssm_verify_kernel[(triton.cdiv(V, BV), batch, HV)](
        q, k, v, a2, b2, A_log, dt_bias, st, rep, out, qsl, si,
        K ** -0.5, 1.0, 20.0, R.NULL_BLOCK_ID,
        q.stride(1), k.stride(1), v.stride(1), a2.stride(0), b2.stride(0),
        st.stride(0), st.stride(1), st.stride(2), st.stride(3),
        rep.stride(0), rep.stride(1), rep.stride(2), rep.stride(3),
        out.stride(1), qsl.stride(0), si.stride(0),
        H=H, HV=HV, K=K, V=V, BK=128, BV=BV, SPEC_QUERY_LEN=T, USE_QK_L2NORM=True,
        num_warps=nw, num_stages=2)


def time_graph(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); scratch.zero_()
        g = torch.cuda.CUDAGraph(); gz = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            scratch.zero_(); fn()
        with torch.cuda.graph(gz, stream=s):
            scratch.zero_()
    torch.cuda.synchronize()
    def t(gr):
        v = []
        for _ in range(50):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); gr.replay(); e1.record(); e1.synchronize(); v.append(e0.elapsed_time(e1) * 1000)
        v.sort(); return v[len(v) // 2]
    return t(g) - t(gz)


res = []
for batch in (1, 4):
    args = inputs(batch)
    tot = batch * T
    ref_out = torch.empty(1, tot, HV, V, device=dev, dtype=torch.bfloat16)
    ref_rep = torch.zeros_like(args[8])
    launch(args, 32, 4, ref_out, ref_rep)
    torch.cuda.synchronize()
    for BV, nw in itertools.product((4, 8, 16, 32, 64), (1, 2, 4, 8)):
        out = torch.empty_like(ref_out); rep = torch.zeros_like(ref_rep)
        try:
            launch(args, BV, nw, out, rep); torch.cuda.synchronize()
        except Exception as ex:
            res.append({"batch": batch, "BV": BV, "nw": nw, "error": str(ex)[:120]}); continue
        same = bool(torch.equal(out, ref_out) and torch.equal(rep, ref_rep))
        us = time_graph(lambda: launch(args, BV, nw, out, rep))
        mib = batch * HV * V * K * 4 / 2**20
        r = {"batch": batch, "BV": BV, "nw": nw, "us": round(us, 2), "GBps": round(mib * 2**20 / us / 1e3, 1),
             "bit_identical": same}
        res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open("/opt/llm/runners/results/verifybench.json", "w"), indent=1)
print("== ALL DONE ==")
