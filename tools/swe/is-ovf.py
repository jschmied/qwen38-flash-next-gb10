"""Exit 0 if the trajectory ended in a context overflow (vLLM or TensorFold wording), else 1. argv: <traj.json>"""
import json, sys
i = json.load(open(sys.argv[1])).get("info", {})
st, msg = i.get("exit_status"), str(i.get("exception_str") or "")
sys.exit(0 if st == "ContextWindowExceededError" or (st == "BadRequestError" and "cache capacity" in msg) else 1)
