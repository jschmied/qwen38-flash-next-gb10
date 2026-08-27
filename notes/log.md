# Log

Newest first. Each entry records what was tried, what happened, and what it ruled out.

## 2026-08-27 — the overlay is abandoned; the target is now a specific deadlock

**Weights complete** (126 GiB, 206/206 safetensors).

**The Python overlay approach is dead.** PR #53899 (which is a strict superset of #53896 —
every file, plus the five `vllm/v1/ple_offload/` modules) has a branch that is inconsistent
with itself: `config/model.py` imports `checkpoint_has_lm_head`, which exists on current
`main` but **not** in that branch's own `transformers_utils/config.py`. GitHub reports the PR
`CONFLICTING`, and reconstructing it would mean guessing which main commit each of 85 files
expects. Not worth it when a built image exists.

**Switched to `vllm/vllm-openai:qwen38-flash-next`.** Verified identical to what both reporters
on vllm#53960 used:

    digest  sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
    vllm    0.1.dev20073+g8e685d198

That matters more than convenience: a result on the same build cannot be waved away as a
different-build artefact. It does mean running a container on a box whose serving stack is
otherwise bare-metal — scoped to this experiment; production is untouched.

**The experiment.** `TP=1`, `VLLM_PLE_CPU_OFFLOAD=1`, **`--enforce-eager`**. The two py-spy
dumps on #53960 show the offload worker idle on an empty queue while the main thread spins in
CUDA-graph `replay`, which reads as a blocking host round-trip captured inside a graph. If that
is the mechanism, disabling graph capture avoids it.

- **Serves** → the fault is graph capture of the PLE path, not the offload mechanism, and there
  is a workaround available today for everyone blocked on it.
- **Still hangs** → the offload mechanism itself is at fault and `--enforce-eager` is eliminated.

Both outcomes get reported to #53960. Nobody has publicly tried this yet (checked before
starting, per the rule this project now follows: check the field before anything expensive).

## 2026-08-26 (later) — someone got there first, by another road

[0xBakeer/qwen38-flash-next-spark](https://github.com/0xBakeer/qwen38-flash-next-spark)
published a working llama.cpp setup the same day. It validates the mmap hypothesis reasoned
about here — the n-gram table is a lookup, never a GEMM, so it can live on NVMe behind the
page cache — and llama.cpp has the primitive to express that directly:

    -ot "per_layer_token_embd=CPU"  -lm mmap

103.7 GiB GGUF (UD-Q4_K_XL), ~76.9 GiB resident, process RSS ~1.4 GiB, ~22 tok/s decode,
~13% degradation from 226 to 19,197 prompt tokens. Load ~3m35s from NVMe.

Three of their findings matter regardless of route:

- **Concurrency crashes**; they run `--parallel 1`. Single-stream only.
- **Speculative decoding bought nothing** — accepted length 2.88, decode stayed ~23 tok/s.
  Their reading: speculation amortises one weight read over k tokens, which does not help
  when the bottleneck is paged embedding lookups.
- **KV is cheap** — 262k context costs ~6 GiB (2 KV heads, mostly linear attention).

**What this does to the case for continuing here.** The headline question — "does it run on a
Spark" — is answered, and the answer is yes at ~22 tok/s, which is *slower than the 27B this
box already serves* (29.5 / 61.6 tok/s on prose / code with DFlash2 at concurrency 16). So
this is not a production upgrade for anyone, and it was never going to be.

What is left open is the part their route cannot reach: concurrency, prefix caching, and a
serving stack that does more than one request at a time. That is the only reason to keep
going with vLLM, and it is a thin reason — worth stating plainly rather than dressing up.

## 2026-08-26 — setup, and the source-build scare

- Model released ~12:29 UTC. `Qwen4ExpForConditionalGeneration`, no vLLM support at
  release; PR #53896 appeared the same day (111 files, unreviewed).
- Sized the builds. NVFP4 (`RadixArk`) at 125.9 GiB is the only one within reach of
  128 GB unified, and still ~9 GiB over what is usable.
- Found the split: main model 78.2 GiB, PLE/n-gram table 47.7 GiB in ten
  `model-plefp8-*` files. Main model alone fits with ~39 GiB to spare.
- Thought a source build was mandatory (`output_gate_type = sigmoid` vs a prebuilt
  kernel defaulting to `silu`). It is not — see
  [why-no-source-build](why-no-source-build.md). This was the decisive finding of the day.
- Prepared the venv (nightly `dev1244+g8d301f075`, chosen because it is 7 commits ahead
  of the PR base and 0 behind, so all 40 modified files match byte-for-byte).
- Staged the 73-file overlay: 0 empty, 0 syntax errors.
- Weights downloading (46% at time of writing).

**Not yet known:** whether the architecture loads at all on sm_121, and whether the
residency gap can be closed. Nothing has been booted.
