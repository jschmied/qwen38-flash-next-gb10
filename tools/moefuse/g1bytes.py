# ncu byte probe (run under: ncu --profile-from-start off --metrics gpu__time_duration.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum ...): L2 miss sectors x 32 B ~ DRAM reads, contention-independent.
import os, sys
os.environ["TRITON_OVERRIDE_ARCH"] = "sm120"
sys.path.insert(0, "/home/jschmied/git/qwen38-flash-next-gb10/tools/moefuse")
import test_moefuse as T, moe_fp4 as mf, torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
M = 3456; E, H, I, TOPK = T.E, T.H, T.I, T.TOPK
x, ids, wts, xq, xs = T.inputs(M, M)
CFGS = [(64, 64, 256, 4, 2, False), (64, 64, 256, 4, 2, True), (64, 128, 256, 8, 2, True), (128, 64, 256, 8, 2, True), (64, 64, 128, 4, 3, True)]
runs = []
for BM, BN, BK, nw, ns, nf in CFGS:
    sid, eid, ntpp = moe_align_block_size(ids, BM, E); NMB = eid.numel(); rows = sid.numel()
    iq = torch.zeros(rows, I // 2, dtype=torch.uint8, device="cuda"); isf = torch.ones(rows, I // 16, dtype=torch.float8_e4m3fn, device="cuda")
    f = lambda sid=sid, eid=eid, ntpp=ntpp, NMB=NMB, iq=iq, isf=isf, BM=BM, BN=BN, BK=BK, nw=nw, ns=ns, nf=nf: mf._gemm1_swiglu_fp4[(NMB * (I // BN),)](xq, xs, T.w13, T.w13_s, sid, eid, ntpp, T.alpha1, T.a2g, iq, isf, M * TOPK, NMB, TOPK=TOPK, H=H, I=I, BM=BM, BN=BN, BK=BK, N_FASTEST=nf, num_warps=nw, num_stages=ns)  # bind per config (late binding ran the last config 5x)
    f(); runs.append(f)
torch.cuda.synchronize()
torch.cuda.profiler.start()
for f in runs: f()
torch.cuda.synchronize(); torch.cuda.profiler.stop()
print("CFGS", CFGS)
