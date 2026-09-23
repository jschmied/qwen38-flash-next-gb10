# HYPOTHESIS: CheckpointTable.touch on 51,200 cold rows = 250-320 ms (touch_bench v2rows 265-285); 64 rows cold <= 3 ms.
import sys, time, json, numpy as np
from concurrent.futures import ThreadPoolExecutor
from vllm.v1.ple_offload.pageable import CheckpointTable
n = int(sys.argv[2]); t = CheckpointTable("/opt/llm/models/qwen38-flash-next-mtpfp4", 320001536, 160, 128)
rows = np.random.default_rng(int(sys.argv[1])).integers(0, t.num_rows, n); pool = ThreadPoolExecutor(64)
t0 = time.perf_counter(); t.touch(rows, pool); print(json.dumps({"rows": n, "ms": round((time.perf_counter() - t0) * 1e3, 2)}))
