DRAFT — needs the user's go. GitHub vllm-project/vllm PR #55122 (2026-09-07).

## The ≥128 KiB filtered path: measured on H100 and A100, and partly fixed

I flagged this path as untested rather than leave it for review to find. It is tested now. Rented an
H100 and an A100 and ran both arms built from source on the box — this branch and upstream at the
merge-base `d9105ea8` — so these are ratios, not absolutes. Three starts per arm, 5 × 50 launches
per cell. Harness and raw output:
[`bench/h100-filtered`](https://github.com/jschmied/qwen38-flash-next-gb10/tree/9646330c13ccdb4b80900babaf19a9517cfb874a/bench/h100-filtered).

**Correctness holds on both parts.** 48 shapes (rows 48/64 × n {4k, 8k, 20k, 40k} × k {512, 2048} ×
{random, tie-heavy, all-equal}), every run: this branch is bit-identical over 6 calls and exactly
equal to the reference. Upstream is non-reproducible on every shape and loses the selected **set** on
every tie-heavy and all-equal case. The bug is present on Hopper and Ampere, not only on the 99 KiB
part I found it on.

**Cost was worse than the rest of the PR, and one cause is now fixed.** `det_select_row` caches the
row in shared memory and otherwise re-reads it from global memory on each of its four radix passes.
The filtered path asked for a compile-time 128 KB — the size of the two candidate buffers this PR
deletes — which caches rows only up to ~32K keys, while A100 offers 163 KiB and H100 227 KiB. The new
commit sizes the request from `cudaDevAttrMaxSharedMemoryPerBlockOptin` minus the kernel's static
`__shared__`.

| 64 rows, k=512 | before | after |
| --- | --- | --- |
| H100 n=20,000 | 2.13–2.24 | unchanged (already cached) |
| H100 n=40,000 | 2.41–2.44 | **1.73–1.75** |
| A100 n=40,000 | 2.63–2.66 | **1.99–2.02** |
| n=65,536 | 2.6–3.0 | unchanged (256 KB of keys cannot be cached) |

Both non-effects were predicted from `det_select_row_bytes` before the run, which is what makes this
a mechanism rather than a correlation. GB10 cannot reach this path and shows 0 of 43 cells moving
against a 6-start baseline.

**What is still open, honestly.** At n=65,536 the path is 2.2–3.0× upstream and no shared-memory
sizing can help: the row cannot be cached on any current part. I built the obvious fix — compact the
threshold bin's survivors in pass 1 so passes 2–3 read ~n/256 instead of rescanning — and it works
(H100 n=65,536: 2.70 → 2.25, A100 2.94 → 2.56). **I am not proposing it**, because it costs 9–15 % on
*cached* rows, and gating it on the cache flag does not remove that: three GB10 cells still regress
with the path compiled in but switched off, at identical registers and static shared memory. The
regressing shapes are `rows=1, n=4096/8192` — decode. Paying decode to speed up long-context prefill
is the wrong trade, so it stays out until someone can make the disabled path free. The patch is in
the repo above if anyone wants to take it further.

So the filtered path is roughly **1.1–2.7×** upstream, worst at very long rows, in exchange for
reproducibility it does not currently have at all. The affected regime is `rows > 32` — large batch
and prefill, not the c=1 decode shape QSA runs. I would rather you weigh that with the numbers in
front of you than discover it after merge.

Happy to run more shapes on either part; the harness makes it minutes.
