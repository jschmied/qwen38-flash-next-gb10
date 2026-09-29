"""Cherry-pick of vllm#58449 (head 848dbf89ec, bilikaz): the QSA side-cache builder updates draft-step metadata in place,
so the V2 speculator runs its fused multi-step draft loop instead of rebuilding metadata between steps. Only
qsa_cache.py (the PR's test file is not installed). Adds a once-per-process witness line in the new update method.
Usage: python patch_58449.py <site-packages>/vllm [off]; or VLLM_QSA_PY=<file> python patch_58449.py [off] (armrun);
VLLM_X_PY=<copy of qsa_cache.py> for a dry run. Backup
*.orig-58449."""
import os, shutil, subprocess, sys, tempfile
args = [a for a in sys.argv[1:] if a != "off"]; off = "off" in sys.argv[1:]
# armrun's source_toggle calls `python patch_58449.py [off]` with the target file in VLLM_QSA_PY
F = (os.environ.get("VLLM_X_PY") or os.environ.get("VLLM_QSA_PY")
     or os.path.join(args[0], "models/qwen4_exp/common/qsa_cache.py"))
DIFF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pr58449-qsa_cache.diff")
M = "# FN58449"
if off:
    shutil.copy2(F + ".orig-58449", F); print("58449 off", M in open(F).read()); sys.exit(0)
if M in open(F).read():
    print("already on"); sys.exit(0)
with tempfile.TemporaryDirectory() as d:
    tgt = os.path.join(d, "vllm/models/qwen4_exp/common"); os.makedirs(tgt)
    shutil.copy2(F, os.path.join(tgt, "qsa_cache.py"))
    subprocess.run(["patch", "-p1", "-s", "-i", DIFF], cwd=d, check=True)
    s = open(os.path.join(tgt, "qsa_cache.py")).read()
a = "    def update_draft_decode_metadata(self, metadata: QSAForwardMetadata) -> None:\n"
assert s.count(a) == 1, s.count(a)
b = (a + "        if not getattr(QSAMetadataBuilder, \"_fn58449_logged\", False):  " + M + "\n"
     "            QSAMetadataBuilder._fn58449_logged = True  " + M + "\n"
     "            from vllm.logger import init_logger as _fn_il  " + M + "\n"
     "            _fn_il(__name__).info(\"FN58449 fused QSA draft metadata update ran\")  " + M + "\n")
s = s.replace(a, b)
compile(s, F, "exec")
if not os.path.exists(F + ".orig-58449"):
    shutil.copy2(F, F + ".orig-58449")
open(F, "w").write(s)
print("58449 on", s.count(M))
