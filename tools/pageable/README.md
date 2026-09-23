# Pageable PLE prototype (findings 225, 226)

The GPU reads the Qwen3.8-Flash-Next PLE n-gram table straight from read-only file mappings. This works on GPUs with
pageable memory access (GB10: `CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES=1`). A CPU thread
prefetches each step's rows, because GPU faults on non-resident file pages are serviced one page at a time.

- `pageable.py` → `vllm/v1/ple_offload/pageable.py`. v1 = a contiguous copy (`VLLM_PLE_PAGEABLE_FILE`).
  v2 = the checkpoint's own safetensors, with a Triton gather through a shard-address table (`VLLM_PLE_PAGEABLE=checkpoint`).
- `venv-hooks.diff`: env-gated hooks in `ple_layer.py`, `v1/worker/gpu/model_runner.py`, and `weight_utils.py`
  (loader skip). Against **vllm 0.28.1rc1.dev524 + our #53899 overlay** (`vllm-venv-fnmain3`), NOT main.
- `v2_kernel_test.py`: bit-exactness against the v1 file, graph replay, warm latency.
- `build_ple_file.py`, `pageable_probe.py`, `pageable_prefetch.py`, `ttft_real.py`: v1 tools and probes.

Status: prototype. Not yet ported onto main's #54371 embedding hierarchy. See notes/prefill-investigation.md 225/226.
