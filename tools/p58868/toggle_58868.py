"""Apply (default) or remove ('off') vllm#58868 (fault mmap'd weight pages on integrated GPUs before the H2D copy)
in the venv whose file armrun passes in VLLM_PKG. Marker: 'def copy_weight_' in model_executor/utils.py."""
import os, subprocess, sys
off = len(sys.argv) > 1 and sys.argv[1] == "off"
f = os.environ["VLLM_PKG"]; sp = f[: f.rindex("/vllm/")]
diff = "/opt/llm/runners/p58868/vllm-only.diff"
on_now = "def copy_weight_" in open(os.path.join(sp, "vllm/model_executor/utils.py")).read()
if on_now == (not off):
    print("already", "off" if off else "on"); sys.exit(0)
args = ["patch", "-p1", "-s", "-d", sp, "-i", diff] + (["-R"] if off else [])
subprocess.run(args + ["--dry-run"], check=True); subprocess.run(args, check=True)
subprocess.run(["find", os.path.join(sp, "vllm/model_executor"), "-name", "__pycache__", "-prune", "-exec", "rm", "-rf", "{}", "+"])
print("58868", "off" if off else "on")
