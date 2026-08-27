# RadixArk NVFP4 on vLLM, one DGX Spark — working

**Status: coherent, correct, concurrent.** Confirmed 2026-08-27 after re-fetching two corrupt
shards. This corrects the published checkpoint tables that list
`RadixArk/Qwen3.8-Flash-Next-NVFP4` as not loading on vLLM: it loads and serves with a
**one-line gate change**.

## Measured

Container `vllm/vllm-openai:qwen38-flash-next` (vLLM `0.1.dev20073+g8e685d198`), one GB10,
`VLLM_PLE_CPU_OFFLOAD=1`, no speculative decoding, `--max-model-len 8192`.

| | |
|---|---|
| weights resident | **76.61 GiB** (PLE offloaded to host) |
| KV cache | 30.99 GiB (room for far longer context) |
| free on device at startup | 114.3 / 121.63 GiB |
| load time | ~11 min (206 shards, cold) |
| engine init | 150 s (compilation 18 s) |

**Decode, concurrency 1:** 13–17 tok/s (no speculation)

| task | tok | tok/s |
|---|---|---|
| code generation | 225 | 17.1 |
| factual recall | 211 | 17.3 |
| German prose | 139 | 15.5 |

**Concurrency scales near-linearly** — the reason to prefer vLLM here over llama.cpp,
which is limited to `--parallel 1`:

| streams | aggregate | per stream | loss |
|---|---|---|---|
| 1 | 17.3 tok/s | 17.3 | — |
| 2 | **32.4 tok/s** | 16.2 | 6% |

Correctness spot-checks all pass, including the trick question ("17 sheep, all but 9 die" -> 9),
correct iterative Fibonacci, correct Galilean moons, and correct German Rayleigh-scattering
explanation.

## The two changes needed

### 1. One-line gate: accept ModelOpt checkpoints for the FP8 PLE

`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`, `_get_ple_embedding_quant_method()`
rejects anything that is not an `Fp8Config`:

```python
if not isinstance(quant_config, Fp8Config):
    return None
```

RadixArk ships the PLE in *exactly* the format this method implements — F8_E4M3 shards plus one
global BF16 `ngram_embedding.weight_scale` — but the body is NVFP4, so `quant_config` is
`modelopt_fp4` and the gate rejects it. The embedding is then built unquantized and loading dies:

```
ValueError: There is no module or parameter named 'ngram_embedding.weight_scale'
            in Qwen3_8FlashNextNGramEmbedding
```

Accepting `modelopt` / `modelopt_fp4` fixes it. One caveat, worth stating because it fails
*silently*: under PLE CPU offload the GPU-side process must **not** register
`weight`/`weight_scale`. `load_weights()` retains only `_offload_weight_scale`, so a
registered-but-never-loaded `weight_scale` shadows it in `_get_embedding_weight_scale()` and the
lookup dequantizes against an uninitialised value — fluent garbage, no error. The offload worker
owns the real weights. See `scripts/apply-pr53896.sh`.

### 2. `--cap-add=SYS_PTRACE` when running PLE CPU offload in Docker

Not documented anywhere we could find, and it kills the server at boot:

```
RuntimeError: pidfd_getfd: Operation not permitted
  torch/multiprocessing/reductions.py:179 in rebuild_cuda_tensor
  vllm/v1/ple_offload/worker.py:482 in accept_registrations
```

`PleOffloadWorker` hands CUDA tensors to the GPU worker over IPC, and `rebuild_cuda_tensor` needs
`pidfd_getfd`, which a default Docker seccomp/capability set denies. Both workers load all 206
shards happily and *then* the engine dies with an unhelpful
`Engine core initialization failed. Failed core proc(s): {}`.

```bash
docker run --gpus all --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined ...
```

Anyone using vLLM's *official* `VLLM_PLE_CPU_OFFLOAD` inside a container will hit this. It does
not affect the mmap-hook approach (single process, no IPC handoff) or SGLang builds.

## Working invocation

```bash
docker run -d --name fnext --gpus all --ipc=host --shm-size 16g -p 8092:8000 \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /path/to/Qwen3.8-Flash-Next-NVFP4:/model:ro \
  -v $PWD/ple_layer.py:/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  vllm/vllm-openai:qwen38-flash-next \
  /model --served-model-name flashnext --host 0.0.0.0 --port 8000 \
  --max-model-len 8192 --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --gpu-memory-utilization 0.90 \
  --distributed-executor-backend mp \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2]}'
```

`--distributed-executor-backend mp` is required: the uniproc executor hangs at TP=1
([vllm#53960](https://github.com/vllm-project/vllm/issues/53960), fixed upstream in `95dc96d1d012`).
Add `--reasoning-parser qwen3` to stop the reasoning trace leaking into `content`.

## Honest positioning

17 tok/s single-stream is **slower than the field's best**. On the same hardware,
[paragontasx](https://github.com/paragontasx/qwen38-flash-next-dgx-spark) reports 31–50 tok/s on
llama.cpp and [Death-By-Tokens](https://github.com/Death-By-Tokens/Qwen3.8-Flash-Next-180B-on-ONE-DGX-Spark)
~27 tok/s on SGLang, both with speculative decoding and 262K context against our 8K.

What this configuration has that they do not is **concurrency**, measured above at 94%
per-stream efficiency across two streams. Whether that is worth 2-3x less single-stream speed
depends entirely on the workload. No speculative decoding was enabled here; that is the obvious
next lever and would close much of the gap.
