# #55122 perf bench — hypothesis, written 2026-09-24 before any build or run

**Question.** On the current PR head `b2312b2de`, run on the model and hardware the PR targets (Qwen3.8-Flash-Next,
GB10 sm_121, vLLM main): is the deterministic `persistent_topk`
1. faster than the exact-top-k workaround, as k3dani's 21–28 % says, and
2. free relative to stock `persistent_topk` end to end?

Nobody has measured this on the current head. MaCoredroid measured no performance at all. k3dani's number predates
253 upstream commits and the port onto `filtered_topk_row`.

**Arms.**
- `base`: merge base `3df4ae15`, standalone build.
- `pr`: PR head `b2312b2de`, standalone build with identical flags.
- `wheel`: the `_C` in the `1ea7c63f4` nightly. This is a sanity check that `base` ≈ `wheel`.
- `exact`: masked `torch.topk` plus −1 fill. `exact_sorted` also sorts the output ascending, as qsafix2 did.

The server arms are `stock` (wheel `_C`), `pr` and `exact`, all on the prod config: MTP n=3, nodrop, 32k slice,
checkpoint-mapped PLE, prefix cache on.

**Expected ranges.** An outcome outside these ranges means I debug the instrument before believing it.

| cell | expected |
|---|---|
| kernel `base`/`wheel`, all shapes | 0.95–1.05 |
| kernel `pr`/`base`, decode rows 4–64 × n 2k–8k, random | 0.95–1.35 (older revisions measured 1.00–2.45, typically 1.14–1.30) |
| kernel `pr`/`base`, tie-heavy data | 1.0–2.0 (the deterministic tie path) |
| kernel `exact`/`pr`, decode shapes | 1.1–3.0 (k3dani: `pr` 21–28 % faster, i.e. `exact` ≈ 1.27–1.39× `pr`) |
| kernel `exact`/`pr`, prefill rows 4096 | 1.1–4.0 |
| server `pr` vs `stock`: TTFT 8k/30k cold, s/turn, ms/tok | within ±3 % (09-04 on dev401: inside the start-to-start band) |
| server `exact` vs `stock`: TTFT 30k | +2 … +20 % (k3dani measured prefill throughput 87–90 %) |
| server `exact` vs `stock`: ms/tok | 0 … +10 % |

**Void conditions.**
- The arm's path line (`QSATOPK env path=…` and `QSATOPK call path=…`) is missing or names the wrong path.
- The `pr` arm's worker does not map `build-pr-server/_C_det.so`.
- The correctness pre-checks fail: set equality with exact on tie-free data, and bitwise repeatability of `pr` on
  tie data.
