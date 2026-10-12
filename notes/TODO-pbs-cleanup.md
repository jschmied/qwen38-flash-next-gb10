# TODO: PBS cleanup (10.0.0.70:/mnt/bulk) - not urgent

Survey 2026-10-12: 4.0 TB, 1.8 TB used, 2.3 TB free. Nothing deleted yet; decide per item with the user.

| where | size | candidates |
|---|---|---|
| gb10/models | 890 GB | Flash Next has 14 variants, ~560 GB (nvfp4 113, abliterated-nvfp4 113, fp8hc 110, exl3-3.05bpw 79, fnext-lh-experts 61, fnext-lh3 61, mtpfp4-plebf16 53, small heads/drafters): keep prod fp8head + one reference, drop superseded. The 27B zoo (~250 GB) STAYS: vLLM-vs-TensorFold comparisons. Nemotron 17 GB stays. |
| gb10/kolibri-drafter | 503 GB | training data `rec` (verified 2026-10-12, keep while the drafter may resume), checkpoints and old runs beyond the release |
| hf | 284 GB | HF snapshots: check each against gb10/models duplicates |
| gb10/work | 133 GB | old job scratch: list, then drop |
| ds41-archive, tmp, venv-archive | ~11 GB | keep (metadata, logs, the 2026-10-12 vLLM venv tars incl. the patched prod venv) |

Before deleting anything: sha256-listed index of what goes (as fork-branches-deleted-*.tsv), and keep INDEX.md current.
