#!/usr/bin/env bash
# Build lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8 from the RadixArk checkpoint we
# already have, fetching only what actually differs.
#
# lovedheart forked RadixArk and rewrote 4 BF16 body shards to FP8 (attn q/k/v/o
# and GDN projections). 202 of 206 shards are byte-identical by published
# lfs.sha256, so they are HARDLINKED -- zero extra disk -- and only 12.45 GiB is
# downloaded instead of 123.4.
set -uo pipefail
SRC=/opt/llm/models/qwen38-flash-next-nvfp4
DST=/opt/llm/models/qwen38-flash-next-fp8mix
PLAN=/opt/llm/.cache-pr/slice_plan.json
REPO=lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8
mkdir -p "$DST" || exit 1
python3 - "$SRC" "$DST" "$PLAN" <<'PY'
import json,os,sys
src,dst,plan=sys.argv[1:4]
d=json.load(open(plan)); n=0; miss=[]
for f in d["link"]:
    s=os.path.join(src,f); t=os.path.join(dst,f)
    if not os.path.exists(s): miss.append(f); continue
    os.makedirs(os.path.dirname(t),exist_ok=True)
    if os.path.exists(t): os.unlink(t)
    os.link(s,t); n+=1
print(f"  hardlinked {n} files"+(f"; MISSING from source: {len(miss)}" if miss else ""))
if miss: print("   ", miss[:5])
with open("/opt/llm/.cache-pr/slice.aria2","w") as fh:
    for f in d["fetch"]:
        fh.write(f"https://huggingface.co/{os.environ.get('REPO','lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8')}/resolve/main/{f}\n  out={f}\n")
print(f"  aria2 input written for {len(d['fetch'])} files")
PY
