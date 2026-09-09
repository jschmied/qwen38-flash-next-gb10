DRAFT — needs the user's go. MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue #30 (@MichaelS1011), 0 comments (2026-09-09).

Independent confirmation from a different serving stack: **the RadixArk build fits on one GB10 under
vLLM too**, and has been our production checkpoint for weeks. 126 GB on disk, `RadixArk/…-NVFP4`
(ModelOpt 0.46.0, `*.self_attn.*` / `*.linear_attn.*` / `*.ple.*` excluded), single Spark, TP 1.

Our numbers, so they can be compared with yours rather than just asserted — GB10, sm_121, vLLM, MTP
n=3, `gpu-memory-utilization` 0.75, 16 k max len, 3 prompts × 3 repeats:

| | |
| --- | --- |
| decode, c=1 | **21.6 – 26.7 tok/s**, tracking draft acceptance (38 % / 54 % / 54 %, accept-length 2.15 / 2.63 / 2.64) |
| TTFT, 7,528-token prompt | **3.19 s** median (≈ 2,360 tok/s prefill) |
| TTFT, 29,288-token prompt | **12.2 s** median (≈ 2,400 tok/s prefill) |

That brackets your 22.8 tok/s / TTFT 2.19 s closely enough that the two stacks look like the same
machine doing the same work, which is the useful part of the comparison.

**One structural difference worth naming.** You mmap the PLE (47.7 GiB). We run the CPU-offload worker
from vllm-project/vllm#53899, which holds the table in ~48 GB of **pinned** host memory. That difference
has an operational consequence that cost us a run yesterday, and it is the kind of thing that gets
misread as "125.9 GiB does not fit":

> The PLE worker failed at startup with `CUDA error: out of memory` while `free` reported **118 GiB
> available**. The available memory was real but it was ~105 GiB of **page cache** left by a large file
> pass, and a pinned allocation of that size cannot displace it fast enough. `sync; echo 3 >
> /proc/sys/vm/drop_caches` before the server starts, and it loads every time.

So on a box this close to full, *what kind* of free memory you have matters, not just how much. If you
ever see the mmap path get slow or flaky right after a big download or a checksum pass, that is the
same pressure showing up in the milder form.

**The honest caveat on "vLLM does it":** the offload path is **not upstream** — there are no
`vllm/v1/ple_offload/` files in the released wheel, and #53899 is still open. Stock vLLM will not do this
today; we run that branch plus a semaphore fix of our own on top of it. Your SGLang route needs no such
thing, which is a point in its favour.

Relevant to `start.sh:13` beyond the throughput note: because the RadixArk build *does* fit, it is
available as a checkpoint swap, not just as a curiosity. We opened #37 about the shipped
`local-inference-lab` checkpoint quantising the GDN/linear-attention path with a five-month-old ModelOpt
where RadixArk and NVIDIA both exclude it — and a build that fits is what makes testing that hypothesis
against #27 cheap.

*AI assistance was used in preparing this comment; every number above is our own measurement on this box.*
