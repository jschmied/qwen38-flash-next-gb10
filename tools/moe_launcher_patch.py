# Make the grouped-GEMM tile-scheduler arguments env-driven: FN_MOE_SWZ (max_swizzle_size, default 1) and FN_MOE_RASTER (M|N, default N).
import re, sys
p=sys.argv[1]; s=open(p).read()
pat=re.compile(r"scheduler_args\{(\s*\\\n\s*)1, GemmKernel::TileScheduler::RasterOrderOptions::AlongN\};")
n=len(pat.findall(s)); assert n==2, n
s=pat.sub(lambda m: "scheduler_args{"+m.group(1)+"fn_moe_swz(), fn_moe_raster<typename GemmKernel::TileScheduler::RasterOrderOptions>()};", s)
helper='''#include <cstdlib>
// FN: env-driven tile-scheduler arguments for the GB10 experiment (read once per process).
static inline int fn_moe_swz() {
  static const int v = std::getenv("FN_MOE_SWZ") ? std::atoi(std::getenv("FN_MOE_SWZ")) : 1;
  return v;
}
template <typename RO>
static inline RO fn_moe_raster() {
  static const char* e = std::getenv("FN_MOE_RASTER");
  static const RO v = (e && e[0] == 'M') ? RO::AlongM : RO::AlongN;
  return v;
}
'''
anchor='#include "cutlass/gemm/device/gemm_universal_adapter.h"\n'; assert s.count(anchor)==1
s=s.replace(anchor, anchor+helper, 1); open(p,"w").write(s); print("patched", p)
