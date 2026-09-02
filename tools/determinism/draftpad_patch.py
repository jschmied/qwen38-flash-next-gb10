"""Draft-prefill trailing slots: the draft forward spans the target's full query (incl. the
num_rejected slots) but ids/positions were only written for the accepted prefix, so the trailing
slots carry stale data from the previous step and write wrong keys into the drafter's cache.
Fix: copy positions over the FULL span and shift ids over the FULL span (rejected tail = the
target's own rejected tokens), then place the sampled token at the adjusted end. Not env-gated
(Triton kernel); install only for the arm. `off` restores byte-exactly."""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
OLD_IDS = '''    # Shift target_input_ids by one.
    for i in range(1, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
'''
NEW_IDS = '''    # Shift target_input_ids by one.  DRAFTPAD (jschmied 2026-09-02): over the FULL target
    # span (incl. the num_rejected trailing slots), so those slots hold the target's own tokens
    # instead of whatever the previous step left there.
    full_len = query_len + num_rejected
    for i in range(1, full_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < full_len
        input_ids = tl.load(target_input_ids_ptr + query_start + block, mask=mask)
        tl.store(draft_input_ids_ptr + query_start + block - 1, input_ids, mask=mask)
'''
OLD_POS = '''    # Copy positions.
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)
'''
NEW_POS = '''    # Copy positions.  DRAFTPAD: over the FULL target span (see above).
    for i in range(0, full_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < full_len
        target_pos = tl.load(target_positions_ptr + query_start + block, mask=mask)
        tl.store(draft_positions_ptr + query_start + block, target_pos, mask=mask)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if "DRAFTPAD" not in s: print("  draftpad not installed"); raise SystemExit
    s = s.replace(NEW_IDS, OLD_IDS).replace(NEW_POS, OLD_POS); assert "DRAFTPAD" not in s
    open(TARGET, "w").write(s); print("  draftpad REMOVED")
else:
    if "DRAFTPAD" in s: print("  draftpad already installed"); raise SystemExit
    assert s.count(OLD_IDS) == 1 and s.count(OLD_POS) == 1, "anchor"
    open(TARGET, "w").write(s.replace(OLD_IDS, NEW_IDS).replace(OLD_POS, NEW_POS)); print("  draftpad INSTALLED")
