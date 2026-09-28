#!/usr/bin/env python3
"""Index the FusedMoE expert mapping by name (VLLM_MOE_NAME_INDEX=1, default on; 0 disables).

RoutedExperts.load_weights tests every checkpoint tensor against the whole expert mapping with a
substring check (`weight_name not in qual_name`): 1,536+ entries for 512 experts, ~103 us per
tensor, ~30 s over the ~297k expert tensors of Qwen3.8-Flash-Next. Every mapping entry starts with
"experts.", so an entry can only match where the tensor name has "experts."; looking up the name's
slices at those positions in a {length: {weight_name: [entry indices]}} table finds exactly the
same entries. They are visited in the original order, and for fused (3-D) checkpoint tensors only
the first consecutive run, which is where the original loop breaks. Entries that do not start with
"experts." fall back to the full scan. src/test_moe_name_index_cpu.py checks it against the
original loop."""
import ast
import sys

RE = f"{sys.argv[1]}/vllm/model_executor/layers/fused_moe/routed_experts.py"
s = open(RE).read()
old = ("        expert_mapping = self.get_expert_mapping(include_fused=True)\n"
       "        for expert_name, loaded_weight in weights:\n"
       "            qual_name = f\"{self.layer_name}.{expert_name}\"\n"
       "            # Fused expert weights can be identified by their 3D tensors\n"
       "            is_fused = loaded_weight.dim() == 3\n"
       "            matched = False\n"
       "            for param_name, weight_name, expert_id, shard_id in expert_mapping:\n")
assert s.count(old) == 1, f"RoutedExperts.load_weights loop: expected 1, found {s.count(old)}"
s = s.replace(old, (
       "        expert_mapping = self.get_expert_mapping(include_fused=True)\n"
       "        _q38_index = _qwen38_expert_index(expert_mapping)\n"
       "        for expert_name, loaded_weight in weights:\n"
       "            qual_name = f\"{self.layer_name}.{expert_name}\"\n"
       "            # Fused expert weights can be identified by their 3D tensors\n"
       "            is_fused = loaded_weight.dim() == 3\n"
       "            matched = False\n"
       "            for param_name, weight_name, expert_id, shard_id in _qwen38_expert_candidates(\n"
       "                _q38_index, expert_mapping, qual_name, is_fused\n"
       "            ):\n"))
s += '''

# --- qwen38-flash-dgx: index the expert mapping by name (VLLM_MOE_NAME_INDEX=1) ---
# qwen38-moe-name-index-begin
import os as _qwen38_idx_os

_QWEN38_MOE_INDEX = _qwen38_idx_os.environ.get("VLLM_MOE_NAME_INDEX", "1") != "0"


def _qwen38_expert_index(mapping):
    if not _QWEN38_MOE_INDEX or any(not w.startswith("experts.") for _, w, _, _ in mapping):
        return None
    by_len = {}
    for i, (_, w, _, _) in enumerate(mapping):
        by_len.setdefault(len(w), {}).setdefault(w, []).append(i)
    return by_len


def _qwen38_expert_candidates(index, mapping, qual_name, is_fused):
    if index is None:
        return mapping
    hits = []
    p = qual_name.find("experts.")
    while p >= 0:
        for n, table in index.items():
            ids = table.get(qual_name[p : p + n])
            if ids:
                hits.extend(ids)
        p = qual_name.find("experts.", p + 1)
    if not hits:
        return ()
    hits = sorted(set(hits))
    if is_fused:
        run = 1
        while run < len(hits) and hits[run] == hits[0] + run:
            run += 1
        hits = hits[:run]
    return [mapping[i] for i in hits]
# qwen38-moe-name-index-end
'''
ast.parse(s)
open(RE, "w").write(s)
print("routed_experts.py: MoE name index applied OK")
