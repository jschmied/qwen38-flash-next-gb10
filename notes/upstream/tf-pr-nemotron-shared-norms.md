# TF PR draft: Nemotron's add_rmsnorm onto the shared norms + norms fixes (ashhart's #736 notes) - NOT posted

POSTED 2026-10-13 as https://github.com/ashhart/TensorFold/pull/796 (user: "post all ready").

Branch cuda-nemotron-shared-norms (worktree ~/git/tf-nnorm), commits 140ae71e + 3e47fe5c on origin/main c36b536d. Not pushed.
A/B data: /tmp/claude-1000/nab (grid2-*.json, speed.jsonl, rows-*.bin; copy to notes/data before /tmp is wiped).

Title: cuda/norms: one tree a width, IEEE 1/sqrt; Nemotron's add_rmsnorm on the shared kernel

## What this changes

Two commits, following the notes on #736:

1. `cuda/norms`:
   - a bf16 row takes the 16-byte tree by its width alone; a misaligned buffer is refused (`error.Misaligned`)
     instead of running on the 1,024-thread tree, so a row's bits no longer depend on where its buffer sits;
   - `1 / sqrt` by IEEE division and square root (`__fdiv_rn`, `__fsqrt_rn`) instead of `rsqrtf`, in the generic,
     16-byte and gated forms: correctly rounded on every architecture, as Nemotron's own norms already were;
   - `if constexpr` for the wide rows, so nvcc no longer warns "loop is not reachable".
2. Nemotron CUDA: `add_rmsnorm` runs on the shared `add_rms_h`; `tf_nemo_add_rmsnorm` is removed. The `glue` test's
   host reference now follows `add_rms_h`'s order (8 values a chunk, chunk c on thread c % 256, the warp then the
   8-warp butterfly, the group sums' 8-lane tree), so it still compares every bit. `norm-rows` gains Nemotron's
   width (2688).

The shared kernel sums in another order than Nemotron's, at the same precision: Nemotron's replies change at
near-ties, as #736's note expected.

## Receipt

- Environment: TensorFold `c36b536d` (1.0.9) plus these commits; one DGX Spark (GB10, sm_121, CUDA 13); Nemotron 3.5
  Lightning MLX 4-bit.
- Kernels: `tf-cuda-test glue` 1,242 of 1,242 comparisons byte-equal to the host references; `norms` and `norm-rows`
  within float64 tolerance (worst 0.996), every row count's bits equal, a misaligned bf16 row refused. Mutation:
  the cross-warp butterfly run 1, 2, 4 instead of 4, 2, 1 fails `glue` (byte-exact) and passes `norm-rows`
  (within tolerance).
- Kernel by itself, 4,096 random rows at 2688: 105 of 11,010,048 outputs differ from `tf_nemo_add_rmsnorm` (in 68
  rows), h identical; of the differing values Nemotron's is nearer float64 in 46, the shared one in 59, total error
  equal. Time per call: 1 and 4 rows 4.1 us both, 64 rows 6.08 -> 4.10 us, 4,096 rows 404 -> 391 us.
- Served, `--parallel 4`, 84 cells (2 prompts, greedy and 5 sampling settings x 4 seeds, drafted and plain, 96
  tokens, reasoning and content hashed): drafted == plain 42/42 on both builds; 30 of 84 replies equal to `main`'s,
  the rest diverge after a near-tie.
- Teacher-forced, 4 documents x 512 tokens (docs, code, German, chat), against `main`: dNLL +0.0032 (per document
  -0.004 to +0.008), argmax agreement 98.04 %, KL 0.0038. The noise floor, `main` with only its own cross-warp sum
  reversed: dNLL +0.0015 (-0.010 to +0.017), argmax 97.80 %, KL 0.0046. The change sits inside it.
- Speed, A B B A: decode 512 tokens 142.8/143.6 and 143.4/144.3 tok/s (`main`) vs 146.8/147.9 and 147.4/148.3
  (this), +2.7 %; prompt 9.6k tokens 1.322/1.311 s vs 1.303/1.298 s, 35k tokens 5.469/5.367 s vs 5.397/5.365 s.
- Tests run: `zig build test` and `zig build test-cuda-host` (27/27 steps, 345/345 tests). `zig fmt --check` clean.
  `lean_check.py`: 538 findings, as on `main`.
- Not run, and why: Metal (unchanged); sm_86/sm_120 (one GB10 here).

## Checklist (as #746)
