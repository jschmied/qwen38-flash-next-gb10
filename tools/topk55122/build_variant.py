# Build one persistent_topk variant as a standalone extension, same flags as /opt/llm/kernel-det/build_det.py.
#   argv: <src dir> <so name> <op namespace> <build dir>
# The src dir holds the revision's csrc/libtorch_stable/{topk.cu,*.cuh,*.h} with persistent_topk renamed to
# persistent_topk_det; the bindings are prod's bindings_det.cpp with the namespace substituted.
import os, sys, torch
from torch.utils.cpp_extension import load
src, so_name, ns, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
b = open("/opt/llm/kernel-det/bindings_det.cpp").read()
assert b.count("(_C_det,") == 2, "bindings anchor"
bind = os.path.join(out, f"bindings_{ns}.cpp")
open(bind, "w").write(b.replace("(_C_det,", f"({ns},"))
arch = os.environ.get("DET_ARCH", "121a")
# The merge base's persistent_topk falls back to top_k_per_row_decode (sampler.cu) when the cooperative launch
# does not fit on a <128 KiB-smem part -- i.e. on GB10. The PR head removed that call.
extra = [os.path.join(src, "sampler.cu")] if os.path.exists(os.path.join(src, "sampler.cu")) else []
load(name=so_name, sources=[os.path.join(src, "topk.cu"), bind] + extra, extra_include_paths=[src],
     extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"],
     extra_cuda_cflags=["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}",
                        "--expt-relaxed-constexpr", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA"],
     build_directory=out, verbose=False, is_python_module=False)
so = os.path.join(out, f"{so_name}.so")
assert os.path.exists(so), so
print(f"built {so} ns={ns} rev={open(os.path.join(src, 'REV')).read().strip()}", flush=True)
