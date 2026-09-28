"""FNSYNCPROBE: env-gated host sync right after speculator.propose() in the V2 runner's sample_tokens.
Usage: python patch_syncprobe.py <site-packages>/vllm [off]. Backup *.orig-syncprobe."""
import os, shutil, sys
pkg = sys.argv[1]; off = len(sys.argv) > 2 and sys.argv[2] == "off"
F = os.path.join(pkg, "v1/worker/gpu/model_runner.py"); M = "# FNSYNCPROBE"
if off:
    shutil.copy2(F + ".orig-syncprobe", F); print("syncprobe off", M in open(F).read()); sys.exit(0)
s = open(F).read()
if M in s: print("already on"); sys.exit(0)
if not os.path.exists(F + ".orig-syncprobe"): shutil.copy2(F, F + ".orig-syncprobe")
a = "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
b = (a + "            if __import__('os').environ.get('FN_SYNCPROBE') == '1':  " + M + "\n"
     "                torch.cuda.current_stream().synchronize()  " + M + "\n"
     "                logger.info_once('FNSYNCPROBE sync after propose')  " + M + "\n")
assert s.count(a) == 1, s.count(a)
s = s.replace(a, b); compile(s, F, "exec"); open(F, "w").write(s); print("syncprobe on", s.count(M))
