# RecoverSSM for Qwen4Exp GDN (local port, not upstream)

Kimi-K3's RecoverSSM speculative-decode protocol, ported to Qwen3.8-Flash-Next's GDN layers. Verify runs from the
checkpoint state and stores a per-token (correction, k, g) record; after sampling, one commit kernel replays the
accepted tokens and writes the fp32 state once. That replaces the four full-state snapshots per step that
`notes/speed-of-light.md` §4o identified as the slow spots. Results: §4s (phase 1), §4t (phase 2).

Base: the prod nightly `1ea7c63f4` plus our prod patches (`vllm-venv-main1ea7`), overlaid onto the clone
`/opt/llm/runtime/vllm-venv-rssm`. Env gate: `FN_GDN_RECOVERSSM=1`; with it unset, the venv behaves stock.

| file | installs to (under `vllm/`) | what |
|---|---|---|
| `recoverssm_gdn.py` | `model_executor/layers/mamba/gdn/recoverssm_gdn.py` | verify kernel + commit kernel + commit context (reuses Kimi's plan/compact kernels) |
| `gdn_recoverssm.py` | `v1/attention/backends/gdn_recoverssm.py` | metadata builder + backend `GDN_RECOVERSSM` |
| `ple_recoverssm.py` | `v1/attention/backends/ple_recoverssm.py` | PLE short conv on the same protocol (phase 2) |
| `patch_rssm.py` | abstract.py, config/vllm.py, qwen_gdn_linear_attn.py, qwen4_exp model.py | phase-1 wiring (mode `none` only) |
| `patch_sticky.py` | config/vllm.py | keeps `use_kda_recoverssm` set when the MTP drafter re-validates the shared cache config |
| `patch_unpack.py` | qwen_gdn_linear_attn.py | two call sites that unpacked exactly two state dtypes |
| `patch_rssm2.py` | abstract.py, config/vllm.py, ple_layer.py | phase 2: align mode + PLE short conv |
| `patch_dbg.py` | abstract.py | one log line per class at KV bind (state count, shapes) |
| `test_recoverssm_gdn.py` | — | kernel test vs the native fused kernel (H=16, HV=48, K=V=128, T=4) |
| `comboprobe.py`, `comboprobe2.py` | — | armrun probes (phase 2 keeps both passes' hashes for the cache-hit check) |
| `spec*.json` | — | armrun specs |

Install order: `patch_rssm.py`, `patch_sticky.py`, `patch_unpack.py`, (`patch_dbg.py`), `patch_rssm2.py`; each takes
the `vllm` package dir as `argv[1]`. Backups in the venv: `*.orig-rssm` (before phase 1), `*.orig-rssm2` (before
phase 2). Limits: V2 model runner, PIECEWISE CUDA graphs only (the builder refuses FULL decode graphs).
