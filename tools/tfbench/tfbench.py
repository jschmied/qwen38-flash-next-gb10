#!/usr/bin/env python3
"""TensorFold's public benchmark (tools/bench_openai.py, unmodified copy, see SOURCE.txt) against our server.
Our test servers require the local bench key, so the header is added in-process (never on argv). Runs their
published command (--tokens 64 --reps 5 --temperatures 1.0,0) and the 400-token variant of their PR #42, and prints
ONE json line: {"arm": ..., "t64": {...}, "t400": {...}} with each cell's median decode tok/s and TTFT."""
import contextlib, json, os, sys, tempfile, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_Req = urllib.request.Request
def _req(url, data=None, headers=None, **kw):
    h = dict(headers or {}); h["Authorization"] = "Bearer sk-bench"
    return _Req(url, data=data, headers=h, **kw)
urllib.request.Request = _req
import bench_openai as B  # noqa: E402
arm = sys.argv[1]; out = {"arm": arm}
for toks in (64, 400):
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        path = f.name
    sys.argv = ["bench_openai.py", "http://127.0.0.1:8092", "flashnext", "--tokens", str(toks), "--reps", "5",
                "--temperatures", "1.0,0", "--output", path]
    with contextlib.redirect_stdout(sys.stderr):  # keep stdout to the one json line
        B.main()
    out[f"t{toks}"] = json.load(open(path)); os.unlink(path)
print(json.dumps(out))
