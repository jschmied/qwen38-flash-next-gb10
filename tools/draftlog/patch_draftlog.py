"""FNDRAFTLOG: log, per single-request verify step, the drafter's top-2 (ids, probs) per draft position, the fed tokens
and the target's greedy token per position, to $FN_DRAFTLOG (JSONL). Measurement only (host syncs). Usage:
  python patch_draftlog.py [off]      (files from VLLM_PKG or the running interpreter's vllm)"""
import os, sys
off = len(sys.argv) > 1 and sys.argv[1] == "off"
pkg = os.environ.get("VLLM_PKG") or os.path.dirname(__import__("importlib.util").util.find_spec("vllm").origin)
if os.path.isfile(pkg):  # armrun's toggle passes the target FILE in VLLM_PKG
    pkg = pkg[: pkg.rindex("/vllm/") + len("/vllm")]
SPEC = os.path.join(pkg, "v1/worker/gpu/spec_decode/speculator.py")
RS = os.path.join(pkg, "v1/worker/gpu/spec_decode/rejection_sampler.py")
M = "# FNDRAFTLOG"
A1 = "            if self.draft_watermarker is not None:\n                sampled = self.draft_watermarker.sample(\n"
B1 = ("            if __import__('os').environ.get('FN_DRAFTLOG'):  " + M + "\n"
      "                from vllm.v1.worker.gpu.spec_decode import rejection_sampler as _fnrs  " + M + "\n"
      "                _fnv, _fni = torch.softmax(logits[:1].float(), -1).topk(2, -1)  " + M + "\n"
      "                _fnrs._FN_TOP2.append((_fni[0].tolist(), _fnv[0].tolist()))  " + M + "\n")
A2 = "        draft_sampled = input_batch.input_ids[input_batch.logits_indices]\n"
B2 = (A2 + "        if __import__('os').environ.get('FN_DRAFTLOG'):  " + M + "\n"
      "            _fn_log(logits, draft_sampled, input_batch)  " + M + "\n")
A3 = "class RejectionSampler"
B3 = ("_FN_TOP2: list = []  " + M + "\n\n\n"
      "def _fn_log(logits, fed, input_batch):  " + M + "\n"
      "    import json, os  " + M + "\n"
      "    top2 = list(_FN_TOP2); _FN_TOP2.clear()  " + M + "\n"
      "    if input_batch.num_reqs != 1 or len(top2) != logits.shape[0] - 1 or logits.shape[0] < 2:  " + M + "\n"
      "        return  " + M + "\n"
      "    rec = {'fed': fed.tolist(), 'tgt': logits.argmax(-1).tolist(), 'top2': top2}  " + M + "\n"
      "    with open(os.environ['FN_DRAFTLOG'], 'a') as f:  " + M + "\n"
      "        f.write(json.dumps(rec) + '\\n')  " + M + "\n\n\n" + A3)
def edit(path, pairs):
    s = open(path).read()
    if off:
        for a, b in pairs:
            assert s.count(b) == 1, (path, "installed block not found", b[:50]); s = s.replace(b, a)
    else:
        if M in s: print("already", path); return
        for a, b in pairs:
            assert s.count(a) == 1, (path, a[:50], s.count(a)); s = s.replace(a, b)
    open(path, "w").write(s); print(("off " if off else "on ") + path, s.count(M))
edit(SPEC, [(A1, B1 + A1)])
edit(RS, [(A2, B2), (A3, B3)])
