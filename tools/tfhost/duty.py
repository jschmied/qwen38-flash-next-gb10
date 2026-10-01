#!/usr/bin/env python3
"""GPU duty cycle and host-API counts from an nsys sqlite export. argv: <sqlite> <generated tokens>
busy = union of kernel + memcpy + memset + graph-replay intervals; window = first GPU start .. last GPU end."""
import json, sqlite3, sys

db, tokens = sqlite3.connect(sys.argv[1]), int(sys.argv[2])
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
iv = []
for t in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET",
          "CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):   # graph-level tracing: one row per replay, kernels inside not listed
    if t in tables:
        iv += db.execute(f"select start, end from {t}").fetchall()
iv.sort()
busy, cur_s, cur_e = 0, None, None
for s, e in iv:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
window = iv[-1][1] - iv[0][0] if iv else 0
kern = db.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
api = dict(db.execute("select s.value, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id "
                      "group by s.value").fetchall())
graphs = sum(v for k, v in api.items() if k.startswith("cudaGraphLaunch"))
syncs = sum(v for k, v in api.items() if "Synchronize" in k)
d2h = db.execute("select count(*) from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2").fetchone()[0] \
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else 0
launches = sum(v for k, v in api.items() if k.startswith("cudaLaunchKernel") or k.startswith("cuLaunchKernel"))
print(json.dumps({"window_ms": round(window / 1e6, 1), "busy_ms": round(busy / 1e6, 1),
                  "duty_pct": round(100 * busy / window, 1) if window else None, "idle_ms_per_tok": round((window - busy) / 1e6 / tokens, 3),
                  "kernels_per_tok": round(kern / tokens, 1), "launch_calls_per_tok": round(launches / tokens, 1),
                  "graph_launches_per_tok": round(graphs / tokens, 2), "syncs_per_tok": round(syncs / tokens, 2),
                  "d2h_per_tok": round(d2h / tokens, 2)}))
