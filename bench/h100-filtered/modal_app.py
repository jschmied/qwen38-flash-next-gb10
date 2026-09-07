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
def bench(arch: str = "90a") -> str:
    import subprocess, sys
    env = {**os.environ, "DET_ARCH": arch, "DET_BUILD_DIR": "/tmp/build",
           "TORCH_CUDA_ARCH_LIST": {"90a": "9.0a", "80": "8.0", "100a": "10.0a"}[arch]}
    build = subprocess.run([sys.executable, "/bundle/build.py"], env=env,
                           capture_output=True, text=True, cwd="/bundle")
    if build.returncode != 0:
        return "BUILD FAILED\n" + build.stdout[-4000:] + "\n" + build.stderr[-8000:]
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
    out = bench.with_options(gpu=gpu).remote(arch)
    print(out)
    if out.startswith("BUILD FAILED") or "ALL DONE" not in out:
        raise SystemExit("run did not complete -- see the output above")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"filtered-{gpu.replace(':', 'x')}.txt")
    with open(path, "w") as fh:
        fh.write(out)
    print("\nsaved:", path)
