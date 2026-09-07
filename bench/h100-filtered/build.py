# Build both arms as standalone torch extensions: _C_det (this PR) and _C_base (upstream at the
# merge-base d9105ea8). No vLLM install needed -- torch + nvcc only.
#   DET_ARCH=90a (H100/H200) | 80 (A100) | 100a (B200)
import os, torch
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(here, "base")
arch = os.environ.get("DET_ARCH", "90a")
cuda = ["-O3", "-std=c++17", f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "--expt-relaxed-constexpr", "-DTORCH_STABLE_ONLY", "-DUSE_CUDA"]

for name, src, binding, inc in (
    ("_C_det", os.path.join(here, "topk_det.cu"), os.path.join(here, "bindings_det.cpp"), [here]),
    ("_C_base", os.path.join(base, "topk_base.cu"), os.path.join(base, "bindings_base.cpp"), [base, here]),
):
    out = os.path.join(os.environ.get("DET_BUILD_DIR", os.path.join(here, "build")), name)
    os.makedirs(out, exist_ok=True)
    load(name=name, sources=[src, binding], extra_include_paths=inc,
         extra_cflags=["-O3", "-std=c++17", "-DUSE_CUDA"], extra_cuda_cflags=cuda,
         build_directory=out, verbose=True, is_python_module=False)
    print("built", name, flush=True)

print("det :", torch.ops._C_det.persistent_topk)
print("base:", torch.ops._C_base.persistent_topk)
