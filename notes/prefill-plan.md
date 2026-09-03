# Prefill plan — make TTFT better (2026-09-03)

Agent work on this box is TTFT-bound (`agentic-speed-is-ttft-bound`, finding 61: MTP does not
pay at 30-token turns). Baseline, served config (prefix cache on, batch 4096/8192, det MoE):

| prompt | TTFT | rate |
| --- | --- | --- |
| 7,503 tok | 3.12 s | ~2,400 tok/s |
| 29,263 tok | 11.25 s | ~2,600 tok/s |

Goal: a measured TTFT reduction at both points with unchanged output (bit-identical under the
deterministic stack, or Δtop-1 ≤ noise vs the reference), each lever A/B'd with 3 starts.

## 0. Owed first (queue is busy until they land)

`mtpgrid0` → correction numbers for #53142/#54076/#53798; `kdetab` → e2e decode/TTFT numbers for
PR #55122; `kpytest` → the PR's test file against `_C_det`. Post after the user's go.

## 1. Profile before touching anything (½ day, box free)

`FN_PROF_DIR` is wired (torch profiler via `--profiler-config`). One start, `/start_profile` →
one 8k request → `/stop_profile`, same for 30k. Table: kernel time by component — GDN chunked
scan (triton, `VLLM_GDN_DECODE_KERNEL` also switches the prefill impl, finding 23), QSA indexer
logits + `persistent_topk`, QSA sparse attention, MoE (FlashInfer CUTLASS, M in the thousands),
dense/attention GEMMs, **PLE gather** (CPU, FP8 table RAM-resident ~51 GB, `VLLM_PLE_CPU_OFFLOAD=1`),
hyperconnections, and idle/CPU gaps. Decision gate: the top three consumers set the order below.
Do not skip this — the last profile was prefill-contaminated (`the-prefill-decode-confound.md`).

## 2. Levers, ordered by expected value ÷ cost

| # | lever | why | cost | risk |
| --- | --- | --- | --- | --- |
| A | **main build** (nightly aarch64 wheel `0.28.1rc1.dev345+g4cc0cb6f7` exists; arch `Qwen4ExpForConditionalGeneration` = our checkpoint) with #54513 (prefill/decode-split QSA indexer, up to 1.3× on the indexer, workspace 128→512 MiB) | biggest merged upstream prefill change since the preview | pip venv clone + the known reinstall casualties (FP8 shim, PLE backport `4e8b849b8d97`, det patches) | preview-only features may be missing; re-validate load defects (`flashnext-fp8mix-checkpoint`) |
| B | **PLE on GPU** instead of CPU gather — HashK-PLE (Death-By-Tokens, 51 → 12.8 GB, trainless) or a plain GPU-resident slice if memory allows | if the profile shows the CPU gather in the prefill critical path, this is the only lever that removes it | download (ask first) + loader work | quality: verify Δtop-1 vs reference; memory headroom at 128 GB |
| C | GDN prefill implementation A/B (`VLLM_GDN_DECODE_KERNEL` triton vs cuda — it switches prefill too) | two implementations exist; never timed for prefill | 2 arms × 3 starts, 40 min | cuda arm hangs at c~32 with FP8 GDN (serve-fnext comment) — test c=1 only |
| D | `--max-num-batched-tokens` on the winner of A/C (4096 / 8192 / 16384) | already measured on the preview: 16384 −7 % at 32k, worse at 4k; re-check on the new stack | 3 arms × 3 starts | none |
| E | MoE prefill backend at large M (`flashinfer_cutlass` vs `marlin`; b12x/cutedsl vetoed on this checkpoint) + autotune-cache warm vs cold | prefill M is thousands; the decode choice may not be the prefill choice | 2 arms | determinism: only `cutlass` has the det finalize |
| F | Kernel-det vs exact top-k (already in `kdetab`) | −4 pp TTFT vs the Python workaround | queued | — |
| G | Prefix-cache block granularity / align cost at 8k (finding: spec decode costs one prefix block) | second-order | later | — |

Not levers (measured or excluded): `cooperative_topk` (needs TMA, excluded on sm_12x); llama.cpp
#28136 (cold-start PLE read from SSD — our table is RAM-resident); MTP depth for TTFT.

## 2b. Field calibration (sweep 2026-09-03, `the-field.md`)

Every GB10 stack — vLLM or SGLang, PLE on CPU / NVMe / swap / **GPU (HashK)** — reports 2.3–2.7k
tok/s cold prefill; an RTX Pro 6000 does 11–13k. So prefill here is compute-bound near the hardware
line and lever **B (PLE on GPU) is not a prefill lever** (airawatraj: 2.4–2.5k with GPU-resident
HashK) — keep it only for memory. Realistic win from A/C/D/E: 5–15 %, unless the profile finds an
idle gap. The multiplier that is real is the warm prefix cache (~10× on repeated prefixes; we
already hold −40 % TTFT after turn 1 and one lost block under spec decode) — so lever G moves up:
maximise hit rate in the agent loop (block size, spec-decode block cost, what the client resends).
Also seen: crimsonjoo's vLLM recipe pays −17…−40 % prefill for its exact top-k default; PR #55122 is
the answer for them (comment after the user's go).

## 3. Method

- One runner per lever, `TTFT=1` arms (ttft.py, 8k + 30k, 3 requests), 3 starts, interleaved.
- Activation line per arm in the log; never two variables in one `Environment=`.
- Quality gate per winner: the deterministic stack makes outputs bit-comparable — compare the
  8k/30k completions against the current stack before adopting anything.
- Record every result as a numbered finding in `determinism-investigation.md`'s successor file
  `prefill-investigation.md`; raw data to `notes/data/`.

## 4. Order of execution

owed measurements → profile (1) → A (main build, because C/D/E should be measured on the stack we
will keep) → C, D, E on it → B if the profile says PLE gather matters → write-up + prod switch
(user decision).

## 5. Ideas that are ours, not the field's (2026-09-03)

Replicating the field buys ≤ 15 %. These are the directions nobody in the sweep is on, ranked by
expected TTFT effect in the agent loop ÷ effort:

1. **Spec decode must not cost a prefix block.** Our own finding (`spec-decode-prefix-cost-agentloop`):
   with MTP on, the agent loop's prefix-cache hit rate drops 69 → 43 %, TTFT ×3 on the turn after,
   one extra cold turn. That is a bug, not a property: the verified tokens + bonus token leave the
   last block partially written or unaligned to the mamba align boundary, and the next turn
   re-prefills it. Bisect with the hit counters we already trust (`prefix_cache_hits_total` deltas,
   never `usage.cached_tokens`), fix in the scheduler/align path, upstream it. Direct TTFT win on
   every agent turn; nobody else measures hit rate per turn.
2. **PLE gather off the critical path.** The n-gram lookup depends only on token ids, so every row
   for a whole prompt is known at tokenization time; on a cache hit the prefix's rows are not needed
   at all. If the profile shows the CPU gather inside prefill time, prefetch the next chunk's rows
   during the current chunk (one-chunk pipeline) and skip rows for cached blocks. Cheap, and the
   worker already streams H2D asynchronously (`ple_offload/worker.py`), so the plumbing exists.
3. **Index sharing across query positions in prefill.** The QSA indexer scores every query token
   against every block (O(T × blocks); at 30k that is the bulk of the indexer cost). vLLM already
   shares one selection across MTP draft positions (`index_share_for_mtp_iteration`); for prefill,
   compute the selection every s-th query position and reuse it for neighbours. A quality trade —
   and we are the only stack that can measure it exactly (bit-comparable outputs, logprob
   divergence per layer). Gate: profile share of the indexer ≥ 20 %.
4. **Persist the hybrid state across restarts.** With bit-exact determinism, GDN state + QSA KV of a
   prefix is a pure function of the tokens: snapshot at align boundaries to host memory/NVMe and
   reload — a KV-connector for hybrid models. Big effort; the payoff is cold-start TTFT of long
   system prompts after every restart/eviction, which every recipe in the sweep pays.

Not novel, dropped: PLE placement, batch sizes, MoE backend sweeps, GDN kernel A/Bs (keep as
5–15 % housekeeping on the main build).
