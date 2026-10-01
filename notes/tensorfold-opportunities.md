# What our findings offer TensorFold (sweep 2026-10-01, against TensorFold 0.6.0 = `c464617`)

User: "do a full sweep of our findings repo and check what points would help tensorfold". Three read-only reviews
(decode + speculation; prefill + memory + PLE; numerics + quant + eval + the field), each checked against TensorFold's
source. The four most actionable claims re-verified by hand (EXL3 gather, `release()` + table file, seed fallback,
draft-vocab plumbing). Nothing here is measured on TensorFold unless it says so.

## Ranked

| # | Item | Our evidence | TensorFold 0.6.0 | Expected | Effort |
|---|---|---|---|---|---|
| 1 | **Thread + row-view the EXL3 n-gram gather** | finding 226: flat byte-offset gather 3.59–3.62 s vs 2-D row views 0.21–0.30 s for 51,200 cold rows; finding 225: 480k rows 2.8 s at 32 threads, 1.6 s at 64, serial faults 88 s | `exl3_pack.py:156-170`: one thread, `maps[f][at[:,None]+cols]`; mapped without `MADV_RANDOM` (`:144`); the BF16/FP8 table already uses row views (`host_table.py:168`) | prompt TTFT on EXL3 (our 0.5.0 gap: 9.3 s vs 2.6 s at 8k) | small, same bytes |
| 2 | **Queue cold-page reads before the decode gather** (WILLNEED + POPULATE_READ) | §4x/§4z, vllm#58835: 57 cold pages 4.6 → 0.36 ms; server −2.35/−2.91 ms/step cold, majflt 29–33 → 0.1/step, 48/48 hashes | not done: decode rounds gather < 1,024 rows on one thread (`host_table.py:96-105`); 0.6.0 lets KV growth evict the table (`multi.py:72-73`), so cold pages recur under `--parallel` | ~2–3 ms a round while cold | ~40 lines |
| 3 | **Read prompt rows ahead on CUDA, more gather threads** | prefill plan §5.2; finding 225: prefetch 12–17 ms/step hidden in a ~1.3 s chunk | `ReadAhead` exists but only the Mac path uses it (`host_table.py:310-338`); CUDA `stage()` gathers synchronously (`forward.py:420-447`); `GATHER_THREADS=16` | gather off the TTFT path | small |
| 4 | **Prompt chunk 2048 → 4096/8192 (with grouped raster in `_b16mm`)** | prefill-batch-size.md: 4096 −13 % vs 2048 at 28k; finding 90: MoE 4.03 → 2.47 → 1.76 µs/token at 2k/4k/8k; findings 149/150: ungrouped raster loses 2–3× at N ≥ 3840 and ≥ 14 MiB activations | `PREFILL_ROWS=2048` (`decode.py:115`, `exl3.py:17`); `bf16.py:56,109` M-fastest, ungrouped (others group) | TTFT −10…−15 % (est.) | constant + sweep; raster small |
| 5 | **Server-side seed salt for evals** | §5aj: rerun = replay, 48/58 patches byte-identical; §5al needed `seed: 2` per request | `cuda/server.py:290` falls back to `seed_for(prompt)`; the salt (`exact_sampling.py:36`) is not settable | correct run-to-run spread / pass@k on TF | hours |
| 6 | **Byte-floor ledger + in-model profile tooling** | §2a/§3/§3c/§4o/§5q (floor 45.2 vs 59.2 ms/cycle; dirty L2 out_proj 71 → 99 µs) | none | locates TF's gap to its own floor (serial EXL3 ~35 tok/s) | 1–2 days (`tools/sol`, `tools/prof`) |
| 7 | **32k draft vocabulary for the MTP head** | det-135, 3 starts: 8k/16k/32k all +6.4–6.8 % c=1 vs full head, acceptance −1.9/−1.1/+0.5 pp; §5k coverage 99.0–99.9 % | default 79,591-row draft head (`weights.py:374-384`); `draft_vocab` is a parameter (`weight_types.py:262`) | ~2–3 % (≈1–1.5 ms/cycle) | trivial to try |
| 8 | **Loader guards: refuse, don't cast, unknown quantized bytes** | failure-modes A2b: FP8_PB_WO loaded as BF16 = fluent garbage, no error | `weight_bf16` casts any non-block-FP8 (`weights.py:195-196`); gate accepts NVFP4/W4A16 on any layer (`__init__.py:66`) | prevents silent garbage (e.g. MXFP8 head, W4A16 GDN checkpoints) | hours |
| 9 | **Thai/Devanagari quality canary** | det-196: vLLM NVFP4 5.68 per 1k Thai chars corrupted at 1.0/0.95/20 (llama.cpp 0); the copy canary caught a broken export 0/72 vs 18/72 | none | cheap regression test for packs/FP8G; a ~0 on TF localises the defect to vLLM | small (`data/thaiprobe.py.txt`) |
| 10 | **Per-expert NVFP4 scales checkpoint** | combining-mark-regression: −0.0152 NLL (t = −3.92), Devanagari −0.089, Thai −0.025; block-of-128 rebuild null | TF reads per-expert and separate gate/up scales already (`weights.py:256`, `nvfp4/experts.py:76`) — immune to vllm#54974 | docs pointer + one NLL check | trivial |
| 11 | **EXL3 startup may drop the table pages it prefetches** (code reading, unmeasured) | §4e (load evicts the table) | `NgramTable` reads small tensors via `pk.get` (`exl3_pack.py:151-154`) → file marked touched (`:77`) → `release()` = whole-file `POSIX_FADV_DONTNEED` (`:84-88`, called `exl3.py:204`) while `in_background(table.prefetch)` runs | first-request TTFT after start | trivial if real; check majflt first |
| 12 | **GDN state write before out_proj; GDN shuffle reduction** | §4o dirty-L2 curve (0/3/6/12 MiB → 71/75/82/99 µs); §4u out_proj 101 → 74 µs; cudafast survey: 40 → 6 shuffles −25 % on ds4 | `gdn.cu:150` writes full fp32 state each verify, replays on reject (`forward.py:545`); `gdn.cu:32-46` ~40 shuffle steps | 0.2–0.5 ms/step (unmeasured on TF) | small/medium; profile first |

Also worth a sentence each: agent recipes at 128k context, not 64k (§5al: 15/17 overflows solved at 131k); bench
hygiene in `bench_openai` (64 → 400 tokens moved our number 55.9 → 63.7; report acceptance per prompt); MTP dense
layers bf16 vs trellis in EXL3 packs (§5h/§5i suggest an A/B); int8/int4 KV quality has no numbers on either side.

## Does not transfer / TensorFold already better
QSA top-k ties (TF breaks ties to the lower block, `attention.py:245`); vLLM batch-invariance / cuBLAS M-dependence /
FlashInfer atomics (TF's row-invariant kernels); PLE one-step-behind semaphore (TF's ReadAhead is keyed by ids);
multi-prefill state index (vllm#55375; TF has a test); hyper-connection decode fusion, fused GDN qkv|z|b|a, exact-size
verify graphs, keyed probabilistic drafts, draft head over the prompt, NVFP4 drafter experts (TF has all); L2 swizzle
on its quantized prompt GEMMs (already grouped); RecoverSSM (TF replays from records); vLLM paging artefacts (block
granularity, MTP trailing block, KV pool coupling, prefix-cache nondeterminism); precision cuts (bf16 SSM state, NVFP4
GDN, FP8 KV) conflict with TF's no-precision-for-speed rule — TF already has int8/int4 KV.
