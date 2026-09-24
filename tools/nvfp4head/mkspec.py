import json, sys
cfg = open("/opt/llm/runners/nvfp4head/best.cfg").read().strip()
spec = {
  "name": "nvfp4head", "venv": "/opt/llm/runtime/vllm-venv-main1ea7",
  "model": "/opt/llm/models/qwen38-flash-next-mtpfp4",
  "env": {"FN_MAXLEN": "32768", "FN_SEQS": "16", "FN_SPEC_METHOD": "mtp", "FN_SPEC_N": "3",
          "FN_SPEC_NODROP": "1", "FN_SPEC_LOCALARGMAX": "1",
          "FN_DRAFT_VOCAB": "/opt/llm/runners/dv/draft_vocab_32768.txt",
          "FN_PLE_OFFLOAD": "0", "FN_EXTRA": "--engram-config {\"checkpoint_mapped\":true}"},
  "probe_cmd": ["<venv>/bin/python", "/opt/llm/runners/nvfp4head/decprobe.py", "{arm}"],
  "starts": 3,
  "arms": [
    {"name": "base", "log_must_contain": ["BF16 slice of the FP8 head", "Mapped PLE table"],
     "log_must_not_contain": ["NVFP4 slice, FNNVFP4"]},
    {"name": "nvfp4", "env": {"FN_DRAFT_HEAD_NVFP4": "1", "FN_NVFP4_CFG": cfg},
     "log_must_contain": ["NVFP4 slice, FNNVFP4", "Mapped PLE table"],
     "log_must_not_contain": ["BF16 slice of the FP8 head"]}]}
json.dump(spec, open("/opt/llm/runners/nvfp4head/spec.json", "w"), indent=1)
print("spec written, FN_NVFP4_CFG =", cfg)
