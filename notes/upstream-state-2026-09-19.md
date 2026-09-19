# Upstream survey, 2026-09-19: Flash-Next is mainstream and our overlay stack is superseded

## The headline

**`#53896 "[Model] Support Qwen3.8-Flash-Next"` is MERGED.** Upstream main carries a 34-file
`vllm/models/qwen4_exp/` tree with both backends:

    common/{ple,hyperconnection,qsa_cache}.py     config.py
    nvidia/{model,model_state,mtp,indexer_qsa,hyperconnection,low_latency_gemm,ngram_embedding}.py
    nvidia/ops/{ple,qsa,hc}.py
    amd/{model,model_state,mtp,indexer_qsa,hyperconnection,low_latency_gemm,ple_layer,qsa}.py
    amd/ops/{ple,qsa,hc}.py
    vllm/model_executor/warmup/qwen4_exp_qsa_warmup.py

So the model we have been carrying as a private overlay now has an official implementation, including
its own PLE. **That PLE is a different one from ours** — and separately, the `ple_offload` symbols in
upstream belong to **Gemma** (`gemma3n_mm.py`, `gemma4_mm.py`, `gemma4_unified.py`), which is a
different feature that merely shares the name. Both facts matter when reading old notes.

## Our upstream items, actual state

| item | state | consequence |
| --- | --- | --- |
| #53896 Support Qwen3.8-Flash-Next | **MERGED** | the whole private overlay stack is superseded |
| #55375 fix state index strides in fused PLE conv | **MERGED** | our local stride patch is redundant |
| #55180 SM 12.x blockwise FP8 CTA swizzle | **MERGED** | redundant |
| #53899 Support PLE-Offload for Flash-Next | OPEN | a feature on top of the merged base |
| #55430 tile-union QSA | OPEN, active 2 d ago | |
| #55122 persistent_topk determinism | OPEN, active 1 d ago | see #56346 below |
| #54948 MoE fused-finalize env flag | OPEN | |
| #54912 QSA raw-key ring widen | OPEN | |
| #55467 our PLE stride PR | CLOSED | closed by us as a duplicate of #55375 |
| #55661, #55174, #54367, #53596 | CLOSED | |

Worth watching: **#56346 "[Perf][Kernel] Add sampled filtering for persistent top-k"** is merged and
touches the same kernel as our open #55122.

## Version gap

Upstream main is `proto-v0.3.0-147-g133b71e0b`, with `v0.30.0rc1`/`rc2` tagged. Our venvs:

| venv | vLLM | behind main |
| --- | --- | --- |
| `vllm-venv-fnmain3` | 0.28.1rc1.dev524 | **653 commits** |
| `vllm-venv-027` | 0.27.1 | — |
| `vllm-venv-main-dflash2` | 0.26.1rc1.dev1049 | **1,393 commits** |
| `vllm-venv-fnext` | 0.1.dev20073 | ancient |

## What this implies

The env move is no longer "rebase our patches onto a newer vLLM". It is **drop the overlay and use the
upstream model**, then re-establish what we measured — because the upstream implementation is a
different one and none of our numbers transfer to it without re-measurement.

Open question before any of that: does the restored `qwen38-flash-next-fp8head` checkpoint load under
upstream's `qwen4_exp`? Our checkpoint was produced for the private implementation, and the config
class (`vllm/models/qwen4_exp/config.py`) may expect different key names.

## A search that lied

`git log --grep=PLE -i` matched **examPLE, multiPLE, samPLEd, comPLEte, supPLEmental** and returned a
page of unrelated commits. Case-insensitive substring search on a three-letter token is useless; the
answer came from `git grep -l ple_offload -- vllm/` and `git ls-tree`.
