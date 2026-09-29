"""FNKSTOP overlay (confidence-stopped drafting, speed-of-light §5ah). Installs spec_decode/fn_kstop.py and hooks:
  spec_decode/speculator.py               update() after each draft sample (probabilistic-drafting path)
  spec_decode/autoregressive/speculator.py set_nreq() in propose(); IF-node predicate on the drafter's decode manager
  cudagraph_utils.py                      capture inside the IF node when the manager carries a predicate
  spec_decode/utils.py                    d_max copy after drafting; [-1] * d_max drafts to the scheduler
Everything is inert unless FN_KSTOP=1. Usage: python patch_kstop.py <site-packages>/vllm [off]. Backups *.orig-kstop.
VLLM_X_ROOT=<dir with a copy of the vllm tree> patches the copy instead (dry run)."""
import os, shutil, sys

pkg = os.environ.get("VLLM_X_ROOT") or sys.argv[1]
off = len(sys.argv) > 2 and sys.argv[2] == "off"
M = "# FNKSTOP"
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_kstop.py")
MOD = os.path.join(pkg, "v1/worker/gpu/spec_decode/fn_kstop.py")
IMP = "from vllm.v1.worker.gpu.spec_decode import fn_kstop as _fn_kstop  " + M + "\n"
LOG = "from vllm.logger import init_logger\n"

EDITS = {
    "v1/worker/gpu/spec_decode/speculator.py": [
        (LOG, LOG + IMP),
        ("            if self.draft_watermarker is not None:\n",
         "            if _fn_kstop.ENABLED:  " + M + "\n"
         "                _fn_kstop.update(logits, draft_step)  " + M + "\n"
         "            if self.draft_watermarker is not None:\n"),
    ],
    "v1/worker/gpu/spec_decode/autoregressive/speculator.py": [
        (LOG, LOG + IMP),
        ("        max_query_len = input_batch.num_scheduled_tokens.max()\n",
         "        max_query_len = input_batch.num_scheduled_tokens.max()\n"
         "        if _fn_kstop.ENABLED:  " + M + "\n"
         "            _fn_kstop.set_nreq(num_reqs, self.max_num_reqs, self.device)  " + M + "\n"),
        ("            decode_query_len=1,\n        )\n",
         "            decode_query_len=1,\n        )\n"
         "        if _fn_kstop.ENABLED:  " + M + "\n"
         "            self.decode_cudagraph_manager._fn_if_pred = _fn_kstop.pred(  " + M + "\n"
         "                self.max_num_reqs, self.device)  " + M + "\n"),
    ],
    "v1/worker/gpu/cudagraph_utils.py": [
        ("                        with torch.cuda.graph(\n"
         "                            graph, self.pool, stream=self._capture_stream(desc)\n"
         "                        ):\n"
         "                            forward_fn(CUDAGraphMode.NONE)\n",
         "                        _fn_pred = getattr(self, \"_fn_if_pred\", None)  " + M + "\n"
         "                        with torch.cuda.graph(\n"
         "                            graph, self.pool, stream=self._capture_stream(desc)\n"
         "                        ):\n"
         "                            if _fn_pred is not None:  " + M + "\n"
         "                                graph.begin_capture_to_if_node(_fn_pred)  " + M + "\n"
         "                            forward_fn(CUDAGraphMode.NONE)\n"),
        ("                            get_offloader().join_after_forward()\n",
         "                            get_offloader().join_after_forward()\n"
         "                            if _fn_pred is not None:  " + M + "\n"
         "                                graph.end_capture_to_conditional_node()  " + M + "\n"),
    ],
    "v1/worker/gpu/spec_decode/utils.py": [
        (LOG, LOG + IMP),
        ("        self.num_draft_tokens = draft_tokens.shape[1]\n",
         "        self.num_draft_tokens = draft_tokens.shape[1]\n"
         "        if _fn_kstop.ENABLED and _fn_kstop.SIZE:  " + M + "\n"
         "            _fn_kstop.copy_d_async()  " + M + "\n"),
        ("            draft_token_ids = [[-1] * self.num_draft_tokens for _ in self.req_ids]\n",
         "            _fn_n = (_fn_kstop.host_d(self.num_draft_tokens)  " + M + "\n"
         "                     if _fn_kstop.ENABLED and _fn_kstop.SIZE else self.num_draft_tokens)  " + M + "\n"
         "            draft_token_ids = [[-1] * _fn_n for _ in self.req_ids]  " + M + "\n"),
    ],
}

if off:
    for rel in EDITS:
        f = os.path.join(pkg, rel)
        shutil.copy2(f + ".orig-kstop", f)
    if os.path.exists(MOD):
        os.remove(MOD)
    print("kstop off", sum(M in open(os.path.join(pkg, r)).read() for r in EDITS))
    sys.exit(0)

new = {}
for rel, edits in EDITS.items():
    f = os.path.join(pkg, rel)
    s = open(f).read()
    if M in s:
        print("already on:", rel); sys.exit(0)
    for a, b in edits:
        assert s.count(a) == 1, (rel, a[:60], s.count(a))
        s = s.replace(a, b)
    compile(s, f, "exec")
    new[f] = s
for f, s in new.items():
    if not os.path.exists(f + ".orig-kstop"):
        shutil.copy2(f, f + ".orig-kstop")
    open(f, "w").write(s)
shutil.copy2(SRC, MOD)
print("kstop on", {os.path.relpath(f, pkg): s.count(M) for f, s in new.items()})
