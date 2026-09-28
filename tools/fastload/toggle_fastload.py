"""Fast weight loading set for our 1ea7 stack: vllm#58868 (fault mmap'd pages before H2D copies) + blazux patch 16
(FusedMoE expert name index, applies as is) + blazux patch 18 (MTP name prefilter, 1ea7 port). On = back up the 7 files
(*.orig-fastload, once) then apply; 'off' = restore the backups. Marker: '_qwen38_keep_name' in weight_utils.py.
armrun passes a target FILE in VLLM_PKG."""
import os, shutil, subprocess, sys
off = len(sys.argv) > 1 and sys.argv[1] == "off"
f = os.environ["VLLM_PKG"]; sp = f[: f.rindex("/vllm/")]; R = "/opt/llm/runners/fastload"
FILES = ["model_executor/model_loader/weight_utils.py", "models/qwen4_exp/nvidia/mtp.py",
         "model_executor/layers/fused_moe/routed_experts.py", "model_executor/layers/linear.py",
         "model_executor/layers/vocab_parallel_embedding.py", "model_executor/parameter.py", "model_executor/utils.py"]
wu = f"{sp}/vllm/model_executor/model_loader/weight_utils.py"
on_now = "_qwen38_keep_name" in open(wu).read()
if on_now == (not off):
    print("already", "off" if off else "on"); sys.exit(0)
if off:
    for p in FILES:
        shutil.copy2(f"{sp}/vllm/{p}.orig-fastload", f"{sp}/vllm/{p}")
else:
    for p in FILES:
        if not os.path.exists(f"{sp}/vllm/{p}.orig-fastload"):
            shutil.copy2(f"{sp}/vllm/{p}", f"{sp}/vllm/{p}.orig-fastload")
    subprocess.run(["patch", "-p1", "-s", "--no-backup-if-mismatch", "-d", sp, "-i", "/opt/llm/runners/p58868/vllm-only.diff"], check=True)
    subprocess.run([sys.executable, f"{R}/patch_moe_name_index.py", sp], check=True)
    subprocess.run([sys.executable, f"{R}/pf18_1ea7.py", sp], check=True)
subprocess.run(["find", f"{sp}/vllm/model_executor", f"{sp}/vllm/models/qwen4_exp", "-name", "__pycache__",
                "-prune", "-exec", "rm", "-rf", "{}", "+"])
print("fastload", "off" if off else "on")
