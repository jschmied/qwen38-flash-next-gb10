import sys, os
p = os.path.join(sys.argv[1], "config/vllm.py"); s = open(p).read()
if "FNRSSM sticky" not in s:
    old = """        # FNRSSM (jschmied 2026-09-25, local): GDN RecoverSSM for Qwen4Exp, reusing the KDA RecoverSSM plumbing.
        if ("""
    assert s.count(old) == 1
    s = s.replace(old, """        # FNRSSM sticky: derived configs (the MTP drafter's) share cache_config and re-validate with
        # num_speculative_tokens == 0; they must not switch the target's RecoverSSM off again.
        if os.environ.get("FN_GDN_RECOVERSSM", "") == "1" and self.cache_config.use_kda_recoverssm:
            return self
""" + old)
    open(p, "w").write(s); print("patched sticky")
