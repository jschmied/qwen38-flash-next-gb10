"""Swap vllm#58449's qsa_cache.py (head 848dbf89e) into the main venv, or restore the original. Byte-exact:
refuses unless the current file is one of the two known states."""
import hashlib, shutil, sys
T = "/opt/llm/runtime/vllm-venv-main1ea7/lib/python3.12/site-packages/vllm/models/qwen4_exp/common/qsa_cache.py"
D = "/opt/llm/runners/pr58449"
ORIG, PR = f"{D}/qsa_cache.orig.py", f"{D}/qsa_cache.pr.py"
H = {"f399f1b3599beb790cdace6d9100d372df97c0ddf8f4f4668b5ddf3c9d3f111d": "orig",
     "08e4dd2ce6276ef248f201f1311efbfa77823fe3e0548c1df6b56eb6bf93ad86": "pr"}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cur = H.get(sha(T))
if cur is None:
    sys.exit(f"refusing: {T} is in an unknown state ({sha(T)[:12]})")
want = "orig" if sys.argv[1:2] == ["off"] else "pr"
if cur != want:
    shutil.copyfile(ORIG if want == "orig" else PR, T)
print(f"  qsa_cache.py: {cur} -> {H[sha(T)]}")
