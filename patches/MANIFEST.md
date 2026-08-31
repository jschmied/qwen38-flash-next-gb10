# Local vLLM patches this box depends on

Cut against **`0.1.dev20073+g8e685d198`** (the vllm#53896 + #53899 preview build).
`apply.sh` refuses another version: upstream has since renamed the package
`qwen3_8_flash_next` → `qwen4_exp` and refactored `ple_layer.py`, so these would not apply
meaningfully.

| file | what it does | provenance |
|---|---|---|
| `v1/ple_offload/connector.py` | per-request D2H event pool instead of one shared `_input_ready_event` | **upstream** `4e8b849b8d97`, backported |
| `v1/worker/gpu/model_runner.py` | call-site change for the above | upstream, same commit |
| `models/…/common/qsa_cache.py` | widen the QSA raw-key ring to the next legal capacity instead of asserting — unlocks `num_speculative_tokens` 5..8 | ours |
| `models/…/nvidia/qsa.py` | fp8_e4m3 KV on the QSA path — advertises the dtypes | **community**, vllm#54426 gist |
| `models/…/nvidia/ops/qsa.py` | the read side of the same (the part that was missing) | community, same gist |
| `models/…/nvidia/mtp.py` | `quant_config` on the MTP `ParallelLMHead`; draft-config fallback (deep-copied) | ours |
| `models/…/nvidia/hyperconnection.py` | widens `GatedResidual.__init__` to accept `quant_config` and threads it into all three projections (upstream hardcodes `None`) | ours |
| `models/…/nvidia/model.py` | `quant_config` on the body `ParallelLMHead`; passes `quant_config=` to all three `GatedResidual` sites | ours |
| `model_executor/layers/quantization/modelopt.py` | `FP8_PB_WO` + per-channel/per-token dispatch | ours; upstream equivalent in vllm#50617 |

**Order matters, and `apply.sh` hardcodes it.** `hyperconnection.py` must precede `model.py`:
`model.py` passes `quant_config=` at all three `GatedResidual` call sites, and upstream's
`__init__` signature is `(config, use_combine=True, prefix="")`. Applying `model.py` alone
is a `TypeError` at model construction, not a subtle degradation.

**The ring-widening patch needs its `logger` import.** `qsa_cache.py` has no module logger
upstream, and the widening branch is only reachable at n=5..8 — so a missing import would
`NameError` exactly when the patch is doing its job, and pass every test at a legal depth.
`apply.sh` asserts the import is present.

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

`apply.sh` now asserts both of those plus `quant_config=quant_config` in `hyperconnection.py`,
and compile-checks all eight files.

### Two defects fixed 2026-08-31, both found by an audit rather than by a failure

1. **`hyperconnection.py` was missing from the bundle entirely** while being live on the box —
   so the published bundle could not reproduce the machine, and would `TypeError` for anyone
   applying it. This is the exact failure the manifest exists to prevent, recurring one directory
   down.
2. **The compile check never ran.** It used `py_compile`, which writes a `.pyc` beside the source
   regardless of `PYTHONDONTWRITEBYTECODE`; against this venv's root-owned `__pycache__` that
   raises `PermissionError`. Chained with `&&`, a permission error and a real syntax error were
   equally silent. It now uses in-process `compile()`, writes nothing, and was negative-controlled
   against a deliberately broken file before being trusted.

## How this relates to `live/` and `series-*/`

- **`live/`** — full copies of each modified file, a *snapshot* of the box. It is the older
  mechanism and it had drifted: 4 of 5 copies were stale and 4 modified files were absent entirely,
  because it is maintained by hand. Refreshed 2026-08-31.
- **`*.patch` + `apply.sh`** (this bundle) — the *re-application* mechanism, which did not exist
  before. Diffs rather than copies, so re-applying after a vLLM upgrade merges with upstream changes
  instead of clobbering them, and `apply.sh` is idempotent and reports `applied` / `already` /
  `FAILED` per file.
- **`experimental/`** — patches that are *not* applied and not part of the live config. They live
  in a subdirectory precisely so the loop cannot pick them up; see its README.
- **`series-main/`, `series-pr53896/`** — git-format-patch series for our *upstream contributions*,
  a different purpose from either of the above.

The lesson behind the drift is the one worth keeping: a patched venv with a hand-maintained
snapshot is a machine nobody can rebuild. The bundle is generated from the `.pre-*` backups beside
each live file, so it can be regenerated mechanically.
