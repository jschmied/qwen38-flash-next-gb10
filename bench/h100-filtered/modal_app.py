# Run the Filtered-path benchmark on a rented H100 or A100.
#
#   modal run modal_app.py                 # H100  (sm_90a, 227 KiB opt-in smem)
#   modal run modal_app.py --gpu A100-80GB # A100  (sm_80,  163 KiB)
#
# The whole job is a couple of minutes of build plus seconds of kernel time; Modal bills per second.
# Output is printed and also written back to ./results/.
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = {"H100": "90a", "H200": "90a", "A100": "80", "A100-80GB": "80", "A100-40GB": "80", "B200": "100a"}

# torch 2.13.0 + CUDA 13.0, matching the GB10 the bundle was validated on. Do NOT drop the torch
# version: `torch/csrc/stable/accelerator.h` (torch_utils.h:4) does not exist before ~2.10, and
# torch 2.8 fails the build with "No such file or directory" after the image has already been pulled.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install("torch==2.13.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("ninja", "numpy")
    .add_local_dir(HERE, remote_path="/bundle", ignore=["build", "results", "__pycache__"])
)
app = modal.App("topk-filtered-bench", image=image)


@app.function(gpu="H100", timeout=60 * 30)
def bench(arch: str = "90a", routing_ab: str = "0") -> str:
    import subprocess, sys
    os.environ["ROUTING_AB"] = routing_ab
    env = {**os.environ, "DET_ARCH": arch, "DET_BUILD_DIR": "/tmp/build",
           "TORCH_CUDA_ARCH_LIST": {"90a": "9.0a", "80": "8.0", "100a": "10.0a"}[arch]}
    build = subprocess.run([sys.executable, "/bundle/build.py"], env=env,
                           capture_output=True, text=True, cwd="/bundle")
    if build.returncode != 0:
        return "BUILD FAILED\n" + build.stdout[-4000:] + "\n" + build.stderr[-8000:]
    # Second arm: the same build with rows>32 forced through the persistent path.
    if os.environ.get("ROUTING_AB") == "1":
        env["KDET_NO_FILTERED"] = "1"
    run = subprocess.run([sys.executable, "-c",
                          "import torch;"
                          "torch.ops.load_library('/tmp/build/_C_det/_C_det.so');"
                          "torch.ops.load_library('/tmp/build/_C_base/_C_base.so');"
                          "exec(open('/bundle/run.py').read())"],
                         env=env, capture_output=True, text=True, cwd="/bundle")
    return run.stdout + ("\n=== STDERR ===\n" + run.stderr[-4000:] if run.returncode else "")


@app.local_entrypoint()
def main(gpu: str = "H100"):
    arch = ARCH.get(gpu.split(":")[0], "90a")
    out = bench.with_options(gpu=gpu).remote(arch, os.environ.get("ROUTING_AB", "0"))
    print(out)
    if out.startswith("BUILD FAILED") or "ALL DONE" not in out:
        raise SystemExit("run did not complete -- see the output above")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"filtered-{gpu.replace(':', 'x')}{'-persistent' if os.environ.get('ROUTING_AB')=='1' else ''}.txt")
    with open(path, "w") as fh:
        fh.write(out)
    print("\nsaved:", path)


# CPU-only syntax/codegen check. nvcc does not need a GPU to compile, so this costs no GPU
# seconds and -- the point -- no time on the GB10, which is busy benchmarking. Used to keep a
# compile error from burning a queued A/B slot hours later.
#
#   modal run modal_app.py::compile_check --arch 121a
@app.function(timeout=60 * 20)
def compile_check(arch: str = "121a") -> str:
    import subprocess, sys, glob
    inc = subprocess.run([sys.executable, "-c",
                          "import torch.utils.cpp_extension as e;"
                          "print(' '.join('-isystem '+p for p in e.include_paths()))"],
                         capture_output=True, text=True).stdout.strip()
    cmd = (f"nvcc -std=c++17 -O3 -c /bundle/topk_det.cu -o /tmp/topk_det.o "
           f"-arch=sm_{arch} -I/bundle {inc} "
           f"--expt-relaxed-constexpr -DTORCH_STABLE_ONLY -DUSE_CUDA "
           f"-DTORCH_EXTENSION_NAME=_C_det -Xptxas -v")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = [f"$ {cmd}", f"exit={r.returncode}"]
    out.append(r.stderr[-12000:] if r.stderr else "(no stderr)")
    if r.returncode == 0:
        out.append("COMPILE OK")
    return "\n".join(out)


@app.local_entrypoint()
def check(arch: str = "121a"):
    print(compile_check.remote(arch))
