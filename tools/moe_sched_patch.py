# Give each CTA runs of FN_TILE_RUN consecutive tiles instead of every G-th tile (G = grid size), so an expert's tiles stay on
# one CTA and the per-tile tensormap update / pipeline refill happens once per run. Usage: moe_sched_patch.py <header> <RUN>
import sys, re
p, run = sys.argv[1], int(sys.argv[2]); s = open(p).read()
anchor = "  get_current_work_for_linear_idx(uint64_t linear_idx) {\n"
assert s.count(anchor) == 1
body = f'''  get_current_work_for_linear_idx(uint64_t linear_idx) {{
    // FN: contiguous runs of FN_TILE_RUN tiles per CTA (a permutation of the strided order:
    // iteration k of CTA c maps to tile ((k / R) * G + c) * R + k % R). Monotonic per CTA.
    if constexpr (FN_TILE_RUN > 1) {{
      uint64_t const G = total_grid_size_;
      uint64_t const k = linear_idx / G;
      uint64_t const c = linear_idx - k * G;
      linear_idx = ((k / FN_TILE_RUN) * G + c) * FN_TILE_RUN + (k % FN_TILE_RUN);
    }}
'''
s = s.replace(anchor, body, 1)
s = s.replace("#pragma once\n", f"#pragma once\n#ifndef FN_TILE_RUN\n#define FN_TILE_RUN {run}\n#endif\n", 1)
assert "#define FN_TILE_RUN" in s
open(p, "w").write(s); print("patched", p, "RUN=", run)
