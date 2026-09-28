# ncu byte probe (run under: ncu --profile-from-start off --metrics gpu__time_duration.sum,lts__t_sectors_srcunit_tex_lookup_miss.sum ...): L2 miss sectors x 32 B ~ DRAM reads, contention-independent.
import os, sys
os.environ["TRITON_OVERRIDE_ARCH"] = "sm120"
sys.path.insert(0, "/home/jschmied/git/qwen38-flash-next-gb10/tools/moefuse")
import test_moefuse as T, torch
from flashinfer.autotuner import autotune
x, ids, wts, xq, xs = T.inputs(3456, 3456)
with autotune(True):
    T.run_fi(xq, xs, ids, wts, 3456)
T.run_fi(xq, xs, ids, wts, 3456); T.run_tr(xq, xs, ids, wts, 3456); torch.cuda.synchronize()
torch.cuda.profiler.start()
T.run_fi(xq, xs, ids, wts, 3456); T.run_tr(xq, xs, ids, wts, 3456); torch.cuda.synchronize()
torch.cuda.profiler.stop()
