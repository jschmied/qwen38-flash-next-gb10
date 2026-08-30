# Local vLLM patches this box depends on

Cut against **`0.1.dev20073+g8e685d198`** (the vllm#53896 + #53899 preview build).
`apply.sh` refuses another version: upstream has since renamed the package
`qwen3_8_flash_next` → `qwen4_exp` and refactored `ple_layer.py`, so these would not apply
meaningfully.

| file | what it does | provenance |
|---|---|---|
| `v1/ple_offload/connector.py` | per-request D2H event pool instead of one shared `_input_ready_event` | **upstream** `4e8b849b8d97`, backported |
| `v1/worker/gpu/model_runner.py` | call-site change for the above | upstream, same commit |
| `models/…/nvidia/qsa.py` | fp8_e4m3 KV on the QSA path — advertises the dtypes | **community**, vllm#54426 gist |
| `models/…/nvidia/ops/qsa.py` | the read side of the same (the part that was missing) | community, same gist |
| `models/…/nvidia/mtp.py` | `quant_config` on the MTP `ParallelLMHead`; draft-config fallback (deep-copied) | ours |
| `models/…/nvidia/model.py` | `quant_config` on the body `ParallelLMHead` | ours |
| `model_executor/layers/quantization/modelopt.py` | `FP8_PB_WO` + per-channel/per-token dispatch | ours; upstream equivalent in vllm#50617 |

## Why this file exists

Four of these were discovered only by accident during debugging — a stray `print` in
`get_quant_method` was still firing in production. A patched venv with no manifest is a
machine nobody can rebuild.

## Verify after applying

```bash
./apply.sh                       # idempotent; reports applied / already / FAILED
```

Then check the two that fail silently rather than loudly:

- `grep -c _input_ready_event .../v1/ple_offload/connector.py` → **0** (shared event gone)
- `grep -A3 supported_kv_cache_dtypes .../nvidia/qsa.py` → must list `fp8_e4m3`

## How this relates to `live/` and `series-*/`

- **`live/`** — full copies of each modified file, a *snapshot* of the box. It is the older
  mechanism and it had drifted: 4 of 5 copies were stale and 4 modified files were absent entirely,
  because it is maintained by hand. Refreshed 2026-08-31.
- **`*.patch` + `apply.sh`** (this bundle) — the *re-application* mechanism, which did not exist
  before. Diffs rather than copies, so re-applying after a vLLM upgrade merges with upstream changes
  instead of clobbering them, and `apply.sh` is idempotent and reports `applied` / `already` /
  `FAILED` per file.
- **`series-main/`, `series-pr53896/`** — git-format-patch series for our *upstream contributions*,
  a different purpose from either of the above.

The lesson behind the drift is the one worth keeping: a patched venv with a hand-maintained
snapshot is a machine nobody can rebuild. The bundle is generated from the `.pre-*` backups beside
each live file, so it can be regenerated mechanically.
