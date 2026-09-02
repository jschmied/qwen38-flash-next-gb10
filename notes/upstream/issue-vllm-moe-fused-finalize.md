# FILED as https://github.com/vllm-project/vllm/issues/54945 (2026-09-02)

# [Bug] FlashInfer CUTLASS NVFP4 MoE gives different logits for identical requests (fused finalize); `use_fused_finalize=False` is bit-stable

## Your current environment

<details><summary>The output of <code>python collect_env.py</code></summary>

```text
Collecting environment information...
==============================
        System Info
==============================
OS                           : Ubuntu 24.04.4 LTS (aarch64)
GCC version                  : (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Clang version                : Could not collect
CMake version                : version 3.28.3
Libc version                 : glibc-2.39

==============================
       PyTorch Info
==============================
PyTorch version              : 2.13.0+cu130
Is debug build               : False
CUDA used to build PyTorch   : 13.0
ROCM used to build PyTorch   : N/A
XPU used to build PyTorch    : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0] (64-bit runtime)
Python platform              : Linux-6.17.0-1031-nvidia-aarch64-with-glibc2.39
    
==============================
       CUDA / GPU Info
==============================
Is CUDA available            : True
CUDA runtime version         : Could not collect
CUDA_MODULE_LOADING set to   : 
GPU models and configuration : GPU 0: NVIDIA GB10
Nvidia driver version        : 580.173.02
cuDNN version                : Could not collect
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : False

==============================
          CPU Info
==============================
Architektur:                             aarch64
CPU Operationsmodus:                     64-bit
Byte-Reihenfolge:                        Little Endian
CPU(s):                                  20
Liste der Online-CPU(s):                 0-19
Anbieterkennung:                         ARM
Modellname:                              Cortex-X925
Modell:                                  1
Thread(s) pro Kern:                      1
Kern(e) pro Sockel:                      10
Sockel:                                  1
Stepping:                                r0p1
Übertaktung:                             deaktiviert
Skalierung der CPU(s):                   100%
Maximale Taktfrequenz der CPU:           3900,0000
Minimale Taktfrequenz der CPU:           1378,0000
BogoMIPS:                                2000,00
Markierungen:                            fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
Modellname:                              Cortex-A725
Modell:                                  1
Thread(s) pro Kern:                      1
Kern(e) pro Sockel:                      10
Sockel:                                  1
Stepping:                                r0p1
Skalierung der CPU(s):                   100%
Maximale Taktfrequenz der CPU:           2808,0000
Minimale Taktfrequenz der CPU:           338,0000
BogoMIPS:                                2000,00
Markierungen:                            fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti ecv afp wfxt
L1d Cache:                               1,3 MiB (20 Instanzen)
L1i Cache:                               1,3 MiB (20 Instanzen)
L2 Cache:                                25 MiB (20 Instanzen)
L3 Cache:                                24 MiB (2 Instanzen)
NUMA-Knoten:                             1
NUMA-Knoten0 CPU(s):                     0-19
Schwachstelle Gather data sampling:      Not affected
Schwachstelle Ghostwrite:                Not affected
Schwachstelle Indirect target selection: Not affected
Schwachstelle Itlb multihit:             Not affected
Schwachstelle L1tf:                      Not affected
Schwachstelle Mds:                       Not affected
Schwachstelle Meltdown:                  Not affected
Schwachstelle Mmio stale data:           Not affected
Schwachstelle Old microcode:             Not affected
Schwachstelle Reg file data sampling:    Not affected
Schwachstelle Retbleed:                  Not affected
Schwachstelle Spec rstack overflow:      Not affected
Schwachstelle Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Schwachstelle Spectre v1:                Mitigation; __user pointer sanitization
Schwachstelle Spectre v2:                Mitigation; CSV2, BHB
Schwachstelle Srbds:                     Not affected
Schwachstelle Tsa:                       Not affected
Schwachstelle Tsx async abort:           Not affected
Schwachstelle Vmscape:                   Not affected

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.6.17
[pip3] nccl4py==0.4.1
[pip3] numpy==2.2.6
[pip3] nvidia-cublas==13.1.1.3
[pip3] nvidia-cuda-cccl==13.3.3.4.1
[pip3] nvidia-cuda-crt==13.3.73
[pip3] nvidia-cuda-cupti==13.0.85
[pip3] nvidia-cuda-nvcc==13.3.73
[pip3] nvidia-cuda-nvdisasm==13.3.73
[pip3] nvidia-cuda-nvrtc==13.0.88
[pip3] nvidia-cuda-runtime==13.0.96
[pip3] nvidia-cudnn-cu13==9.20.0.48
[pip3] nvidia-cudnn-frontend==1.27.0
[pip3] nvidia-cufft==12.0.0.61
[pip3] nvidia-cufile==1.15.1.6
[pip3] nvidia-curand==10.4.0.35
[pip3] nvidia-cusolver==12.0.4.66
[pip3] nvidia-cusparse==12.6.3.3
[pip3] nvidia-cusparselt-cu13==0.8.1
[pip3] nvidia-cutlass-dsl==4.6.2
[pip3] nvidia-cutlass-dsl-libs-base==4.6.2
[pip3] nvidia-cutlass-dsl-libs-core==4.6.2
[pip3] nvidia-cutlass-dsl-libs-cu12==4.6.2
[pip3] nvidia-cutlass-dsl-libs-cu13==4.6.2
[pip3] nvidia-ml-py==13.610.43
[pip3] nvidia-nccl-cu13==2.30.7
[pip3] nvidia-nvjitlink==13.3.33
[pip3] nvidia-nvshmem-cu13==3.4.5
[pip3] nvidia-nvtx==13.0.85
[pip3] nvidia-nvvm==13.3.73
[pip3] pyzmq==27.2.0
[pip3] tokenspeed-triton==3.8.10.post20260721
[pip3] torch==2.13.0+cu130
[pip3] torch_c_dlpack_ext==0.1.5
[pip3] torchaudio==2.11.0+cu130
[pip3] torchvision==0.28.0+cu130
[pip3] transformers==5.15.1
[pip3] triton==3.7.1
[conda] Could not collect

==============================
         vLLM Info
==============================
ROCM Version                 : Could not collect
vLLM Version                 : 0.1.dev20073+g8e685d198 (git sha: 8e685d198)
vLLM Build Flags:
  CUDA Archs: Not Set; ROCm: Disabled; XPU: Disabled
GPU Topology:
  	[4mGPU0	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	0-19	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

==============================
     Environment Variables
==============================
PYTORCH_NVML_BASED_CUDA_CHECK=1
TORCHINDUCTOR_COMPILE_THREADS=1
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_jschmied
```

</details>

Extra: FlashInfer 0.6.17, single NVIDIA GB10 (sm_121, aarch64, 128 GB unified), model
`RadixArk/Qwen3.8-Flash-Next-NVFP4` (hybrid GDN + MoE, 512 experts, top-k 10, modelopt NVFP4).
The venv carries local patches for this model's GDN KV dtype and PLE offload; none touches the
MoE path, and the control run below goes through the same build.

## 🐛 Describe the bug

Identical chat requests at `temperature=0` return different `logprobs`. It is not a race and not
the prefix cache; it is the FlashInfer CUTLASS NVFP4 MoE's fused finalize, which reduces the top-k
expert outputs with atomics (FlashInfer's docstring for `cutlass_fused_moe(use_fused_finalize=True)`
calls it out as nondeterministic). vLLM never passes the argument, so there is no way to turn it off:
`vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` calls
`flashinfer_cutlass_fused_moe(...)` without `use_fused_finalize`.

**Serve** (no spec decode, no prefix cache, eager — the minimum that shows it):

```
vllm serve RadixArk/Qwen3.8-Flash-Next-NVFP4 --served-model-name flashnext \
  --max-model-len 8192 --max-num-seqs 16 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --enforce-eager
```

Log line: `Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`.

**Probe** — one prompt, three times, hash the top-20 logprobs of every generated token:

```python
import hashlib, json, urllib.request
PROMPT = ("Write a detailed technical explanation of how a copy-on-write page table works in a "
          "modern operating system kernel, covering fork(), page faults, reference counting, and "
          "the interaction with the TLB. Be thorough and precise.")
for i in range(3):
    body = json.dumps({"model": "flashnext", "temperature": 0, "max_tokens": 4, "logprobs": True,
        "top_logprobs": 20, "messages": [{"role": "user", "content": PROMPT}],
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    d = json.loads(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions", body,
        {"Content-Type": "application/json"}), timeout=300).read())
    sig = [hashlib.sha256("|".join(f"{t['token']!r}:{t['logprob']:.12g}"
           for t in tk["top_logprobs"]).encode()).hexdigest()[:8]
           for tk in d["choices"][0]["logprobs"]["content"]]
    print(i + 1, " ".join(sig))
```

### Before / after

| configuration (3 identical requests, 4 tokens each) | default (`use_fused_finalize=True`) | `use_fused_finalize=False` |
| --- | --- | --- |
| prefix cache off, `--enforce-eager`, no spec | token 1 identical, **token 2 differs — 3 distinct of 3** | **all 4 tokens identical, 1 distinct of 3** |
| prefix cache on (the 55-token prompt is split 52 + 3 by the block-aligned mamba split) | **token 1 already differs — 3 distinct of 3** | **1 distinct of 3** |
| prefix cache on + MTP `num_speculative_tokens=5` | not probed separately | **1 distinct of 3** |
| decode, 8-turn agent loop, 130 tok/turn, c = 1, no spec (one measurement each; run-to-run band for this config is 42.8–47.7) | 43.92 ms/tok | 45.50 ms/tok (+3.6 %, indicative) |

Each "after" row was repeated on a separate server start against the same autotune cache.

Control: `--moe-backend emulation` (dequantised experts, everything else unchanged) is bit-identical
in every configuration, at 51.38 ms/tok (+17 %, one measurement). So every other component of this model — GDN
recurrent state, PLE lookup, hyperconnections, sampler — is reproducible; the divergence sits in
the NVFP4 MoE kernel.

How it was localised: per-module output hashes over the three requests. At decode step 1 all 18
layer-0 submodules are identical, including the router (`mlp.gate`) and the shared expert; the
first and only differing module is `mlp.experts`. Same picture in both prefill chunks with the
prefix cache on. `--no-async-scheduling`, `CUDA_LAUNCH_BLOCKING=1` and a forced sync after the
mamba-align postprocess all leave it diverging. Other NVFP4 backends on this build: `marlin` and
`humming` differ already at the prefill token; `cutlass` crashes at init (illegal memory access);
`triton*` is rejected for NvFP4. `VLLM_BATCH_INVARIANT=1` is not available here (no linear-attention
backend implements `supports_batch_invariance()`).

### The one-line change that fixes it

In `flashinfer_cutlass_moe.py`, pass `use_fused_finalize=False` to `flashinfer_cutlass_fused_moe`.
Suggested surface: a new env var `VLLM_FLASHINFER_MOE_FUSED_FINALIZE` (default `1`, unchanged
behaviour), or tie `False` to `VLLM_BATCH_INVARIANT=1`. I can send the PR (a few lines, no kernel
change) if either shape is acceptable.

### Caveat for FlashInfer ≤ 0.6.17 users

With a populated `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`, switching to the non-fused finalize fails at
engine init with `Invalid gemm2 profile id: 50`: the persistent autotune cache key does not
distinguish the two runners (the fused runner has 40 GEMM2 tactics, the non-fused one 20, so a
cached id 40–59 is rejected). FlashInfer fixed this in `MoERunner.get_cache_key_extras()` from
v0.6.18rc2. On 0.6.17 either point `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` at a fresh directory or
backport that method (7 lines).

### Why it matters

At temperature 0 this model is not reproducible request-to-request, which makes greedy evals,
A/B comparisons of serving flags and any bit-level regression test impossible on this backend.
Related report on the same hardware and build, different symptom: #54173.

---

Disclosure: the investigation tooling and this text were drafted with Claude (Claude Code). All
measurements were run on my hardware and reviewed by me.

## Before submitting a new issue...

- [x] Make sure you already searched for relevant issues, and asked the chatbot living at the bottom right corner of the documentation page.
