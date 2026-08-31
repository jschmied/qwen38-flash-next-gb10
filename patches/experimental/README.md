# Not part of the live configuration

Neither of these is applied on the box, and `apply.sh` deliberately does not touch them.
Both were cut against local `.pre-*` backups, so their diff headers carry absolute paths
and `patch -p1` cannot locate a target — they report `FAILED` even on a correctly
patched machine. They sat in the parent directory until 2026-08-31, where `apply.sh`
globbed them and exited non-zero on a **healthy** tree.

| file | what it tried | status |
| --- | --- | --- |
| `qsa-fused-draft-decode.patch` | fuse the QSA draft decode path | abandoned; `_draft_common_metadata` appears 0× in the live tree |
| `skinny-gemm-sm121-gate.patch` | widen the CUTE-DSL skinny-GEMM gate to sm_121 | never landed; `low_latency_gemm.py` is byte-identical to its `.pre-sm121` backup |

The skinny-GEMM idea is **not dead** — the `M=3` gap in the `(336, 10240)` plan it targets
is still present upstream, and an independent GB10 report measures +6.9% end-to-end from
relaxing that gate. Re-cut it against a real checkout before trying again.
