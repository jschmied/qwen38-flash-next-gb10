"""Per-step acceptance log in the scheduler (CPU side), gated on VLLM_ACC_LOG=1:
ACCLOG req=<id tail> pos=<num_computed_tokens before rollback> drafts=<n> acc=<k> ids=<generated ids>"""
import sys
TARGET = "/opt/llm/runtime/vllm-venv-fnext/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
ANCHOR = "                num_rejected = num_draft_tokens - num_accepted\n"
NEW = ANCHOR + '''                if __import__("os").environ.get("VLLM_ACC_LOG"):  # ACCLOG (jschmied 2026-09-02)
                    print(f"ACCLOG req={req_id[-8:]} pos={request.num_computed_tokens} drafts={num_draft_tokens} acc={num_accepted} ids={list(generated_token_ids)}", flush=True)
'''
s = open(TARGET).read()
if sys.argv[1:] and sys.argv[1] == "off":
    if "ACCLOG" not in s: print("  acclog not installed"); raise SystemExit
    open(TARGET, "w").write(s.replace(NEW, ANCHOR)); print("  acclog REMOVED")
else:
    if "ACCLOG" in s: print("  acclog already installed"); raise SystemExit
    assert s.count(ANCHOR) == 1, "anchor"
    open(TARGET, "w").write(s.replace(ANCHOR, NEW)); print("  acclog INSTALLED (inert unless VLLM_ACC_LOG=1)")
