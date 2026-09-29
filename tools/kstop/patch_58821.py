"""Cherry-pick of vllm#58821 (head 5cf894bf93): draft decode graphs keep query length 1 under dynamic SD (fixes #58692,
ZeroDivisionError in SpeculatorCudaGraphManager._init_candidates). Usage: python patch_58821.py <site-packages>/vllm [off]
(VLLM_X_PY=<file> patches that file instead, for a dry run). Backup *.orig-58821."""
import os, shutil, sys
pkg = sys.argv[1]; off = len(sys.argv) > 2 and sys.argv[2] == "off"
F = os.environ.get("VLLM_X_PY") or os.path.join(pkg, "v1/worker/gpu/cudagraph_utils.py")
M = "# FN58821"
if off:
    shutil.copy2(F + ".orig-58821", F); print("58821 off", M in open(F).read()); sys.exit(0)
s = open(F).read()
if M in s:
    print("already on"); sys.exit(0)
a = ("        if (\n"
     "            speculative_config\n"
     "            and speculative_config.uses_dynamic_speculative_decoding()\n"
     "        ):\n")
b = ("        if (\n"
     "            self.decode_query_len > 1  " + M + "\n"
     "            and speculative_config\n"
     "            and speculative_config.uses_dynamic_speculative_decoding()\n"
     "        ):\n")
assert s.count(a) == 1, s.count(a)
if not os.path.exists(F + ".orig-58821"):
    shutil.copy2(F, F + ".orig-58821")
s = s.replace(a, b); compile(s, F, "exec"); open(F, "w").write(s); print("58821 on")
