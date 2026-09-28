# ncu byte probe (run under: ncu --profile-from-start off --metrics gpu__time_duration.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum ...): L2 miss sectors x 32 B ~ DRAM reads, contention-independent.
import os, sys
os.environ["TRITON_OVERRIDE_ARCH"] = "sm120"
sys.path.insert(0, "/home/jschmied/git/qwen38-flash-next-gb10/tools/moefuse")
import test_moefuse as T, moe_fp4 as mf, torch
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
M = 3456; E, H, I, TOPK = T.E, T.H, T.I, T.TOPK
x, ids, wts, xq, xs = T.inputs(M, M)
CFGS = [(64, 256, 128, 8, 4, False), (64, 256, 128, 8, 4, True), (128, 256, 128, 8, 3, False), (128, 128, 128, 8, 3, False), (64, 128, 128, 4, 3, True)]
runs = []
for BM, BN, BK, nw, ns, nf in CFGS:
    sid, eid, ntpp = moe_align_block_size(ids, BM, E); NMB = eid.numel(); rows = sid.numel()
    iq = torch.zeros(rows, I // 2, dtype=torch.uint8, device="cuda"); isf = torch.ones(rows, I // 16, dtype=torch.float8_e4m3fn, device="cuda")
    y = torch.empty(M * TOPK, H, dtype=torch.bfloat16, device="cuda")
    f = lambda sid=sid, eid=eid, ntpp=ntpp, NMB=NMB, iq=iq, isf=isf, y=y, BM=BM, BN=BN, BK=BK, nw=nw, ns=ns, nf=nf: mf._gemm2[(NMB * (H // BN),)](iq, isf, T.w2, T.w2_s, sid, eid, ntpp, T.alpha2, y, M * TOPK, NMB, H=H, I=I, BM=BM, BN=BN, BK=BK, N_FASTEST=nf, num_warps=nw, num_stages=ns)  # bind per config
    f(); runs.append(f)
torch.cuda.synchronize()
torch.cuda.profiler.start()
for f in runs: f()
torch.cuda.synchronize(); torch.cuda.profiler.stop()
print("CFGS", CFGS)
