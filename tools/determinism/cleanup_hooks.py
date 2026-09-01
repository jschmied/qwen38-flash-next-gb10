#!/usr/bin/env python3
"""Remove the unwanted experiment hooks before the next batch. Idempotent; verifies its result.

Audit (pip RECORD diff, 2026-09-01) found ten modified vllm files. Seven are deliberate (FP8-KV,
hyperconnection, lm_head, ring widening, PLE backport). Three edits are unwanted:
  qsa.py   blocks.sort()            UNCONDITIONAL -- a live confound in every arm since morning
  qsa.py   VLLM_QSA_ROWS_PER_CHUNK  env-gated, bought nothing
  qsa.py   VLLM_QSA_TORCH_TOPK      env-gated, bought nothing, and changes the answer
  model.py print("PROBE lmhead ...") a bare debug print left in production

qsa.py: no backup has FP8-KV without the sort, but .prerowcap == FP8-KV + sort and nothing else,
so the clean file is .prerowcap minus that one line. Verified afterwards against .orig: the only
remaining differences must be the FP8-KV lines.

Usage: cleanup_hooks.py [--dry-run DIR]   (dry-run writes results under DIR, touches nothing)
"""
import os, re, sys, shutil, subprocess

SP = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm"
QSA = f"{SP}/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
MODEL = f"{SP}/models/qwen3_8_flash_next/nvidia/model.py"
dry = "--dry-run" in sys.argv
out = sys.argv[sys.argv.index("--dry-run") + 1] if dry else None
def target(p):
    if not dry: return p
    os.makedirs(out, exist_ok=True); return os.path.join(out, os.path.basename(p))

# ---- qsa.py ----
src = open(QSA + ".prerowcap").read()
lines = src.split("\n")
sort_lines = [i for i, l in enumerate(lines) if "blocks.sort(dim=1).values" in l]
assert len(sort_lines) == 1, f"expected exactly one sort line in .prerowcap, found {len(sort_lines)}"
del lines[sort_lines[0]]
clean = "\n".join(lines)
for bad in ("blocks.sort", "VLLM_QSA_ROWS_PER_CHUNK", "VLLM_QSA_TORCH_TOPK", "EXPERIMENT"):
    assert bad not in clean, f"{bad} still present after cleanup"
assert "_cast_kv_tile" in clean, "FP8-KV support lost -- refusing"
cur = open(QSA).read()
if cur == clean:
    print("  qsa.py: already clean")
else:
    if not dry: shutil.copy2(QSA, QSA + ".prehookclean")
    open(target(QSA), "w").write(clean)
    print(f"  qsa.py: hooks removed{' (dry-run)' if dry else ' (backup .prehookclean)'}")

# ---- model.py ----
m = open(MODEL).read()
pat = re.compile(r"[ \t]*import sys as _sys\n[ \t]*print\(f\"PROBE lmhead quant_config=.*?flush=True\)\n", re.S)
hits = pat.findall(m)
if not hits:
    print("  model.py: PROBE print already gone")
else:
    assert len(hits) == 1, f"expected one PROBE block, found {len(hits)}"
    assert hits[0].count("\n") <= 4, "PROBE block larger than expected -- refusing"
    m2 = pat.sub("", m, count=1)
    assert "PROBE lmhead" not in m2
    if not dry: shutil.copy2(MODEL, MODEL + ".preprobeclean")
    open(target(MODEL), "w").write(m2)
    print(f"  model.py: PROBE print removed{' (dry-run)' if dry else ' (backup .preprobeclean)'}")

# ---- verification: qsa.py vs stock must differ ONLY in FP8-KV lines ----
q = target(QSA) if (dry and os.path.exists(target(QSA))) else QSA
d = subprocess.run(["diff", QSA + ".orig", q], capture_output=True, text=True).stdout
changed = [l for l in d.splitlines() if l[:1] in "<>"]
sus = [l for l in changed if re.search(r"sort|ROWS_PER_CHUNK|TORCH_TOPK|EXPERIMENT|PROBE", l)]
print(f"  verify: {len(changed)} lines differ from stock, {len(sus)} suspicious")
for l in sus: print("    " + l[:110])
if sus: sys.exit("VERIFICATION FAILED")
for p in (QSA, MODEL):
    subprocess.run([sys.executable, "-c", f"compile(open({(target(p) if dry and os.path.exists(target(p)) else p)!r}).read(),'x','exec')"], check=True)
print("  verify: both files compile")
