# TF PR: cuda tree attention + block attention (shared)

POSTED 2026-10-12 as https://github.com/ashhart/TensorFold/pull/746 (user: "post"). Branch cuda-attn-shared, commit
c5936c72 on origin/main 0efbd447, worktree ~/git/tf-tattn.

Title: cuda: tree attention and DFlash block attention for every family, checked against float64

## What this changes

CUDA has no shared attention for drafted rows. A family that drafts a tree on CUDA writes its own attention over the
committed keys plus each row's path, and a DFlash-style drafter writes its own block attention. This adds both as
shared modules, operator-only (no model calls them yet), next to `cuda/tree_round` (#734), which does the verify and
commit step of the same round:

- `zig/kernels/cuda/tree_attention.cu`, `zig/src/cuda/tree_attention.zig`: drafted rows of any number of streams in
  one call. Each row sees its stream's committed keys and then its root-to-row path (`paths()` builds the paths from
  tree-local parents, depth up to 128). Keys run in 2,048-key groups on tensor cores, merged in group order, at head
  dims 128 and 256 with grouped KV heads. A stream's fully committed groups are computed once for all its rows (16
  (row, head) pairs as one mma tile, the same tiles and merge as a row's own block), so every partial keeps its bits
  whether a row runs alone or beside others. The grid is fixed: at most 16 group slots per (row, key head), each
  walking its groups, so a captured graph does not grow with the context.
- `zig/kernels/cuda/block_attention.cu`, `zig/src/cuda/block_attention.zig`: a DFlash-style drafter's block rows
  over their stream's context (each stream's K and V cache through a pointer table, within a sliding window) and then the block itself,
  causal or not. One warp per head on tensor cores.
- `zig/src/core/attention_ref.zig`: one query head's attention in float64 with a per-element tolerance (2^-7 of
  sum p|v| / sum p, plus a bf16 half-ulp), the reference both tests check against.

Users we know of: the Qwen3.8-27B family in #608 (target tree attention at dim 256 and its DFlash2 drafter), and any
CUDA family that drafts trees with #734.

## Receipt

- Environment: TensorFold `0efbd447` (1.0.8) plus this commit; one DGX Spark (GB10, sm_121, CUDA 13). No
  checkpoint for the tests.
- Exactness:
  - `tf-cuda-test tree`: dim 128 (48 query / 4 KV heads) and 256 (24 / 4), several streams with different committed
    lengths and trees, every element within the float64 tolerance (worst 0.47 of it). A stream's rows run alone
    equal their bits in the batch; one shared block walking every (row, group), and a grid sized for a 262k
    context, equal the default grid.
  - `tf-cuda-test block`: windowed contexts then the block, causal and not, block widths 7 and 16, within
    tolerance (worst 0.38). A stream run alone equals its bits in the batch.
- Mutations: each of these fails its test: paths cut at depth 3, the score scale off by 1/16, the window bound
  off by one, the causal mask off by one.
- In a model (fork only, not in this PR): with #608's 27B engine wired to these kernels instead of its Triton
  attention, drafted long prompt: attention 46.1 ms against Triton's 49.0 ms; 16 streams at 32k context, code and
  chat at temperature 0 and 1: every `bench_concurrent --alone --serial` cell exact, throughput within -4..+2 % of
  Triton. Replies equal 0.6.6's except two at bf16 near-ties (an exact top-2 tie, and one ulp), where drafted still
  equals serial. Teacher-forced on 1,023 rows of real text: NLL 0.8058 vs 0.8029, argmax agreement 99.12 %.
- Speed in this PR: none claimed, nothing calls it.
- Tests run: `zig build test` and `zig build test-cuda-host` (44/44 steps, 617/622 tests, 5 skipped as on
  `main`). `zig fmt --check` clean. `lean_check.py`: 539 findings, as on `main`.
- Not run, and why: Metal (unchanged); no model on `main` uses it.

Registration lines in `build/cuda.zig`, `cuda/kernels.zig`, `cuda/root.zig` and `tests/cuda/main.zig` sit where
#734, #736 and #740 add theirs; whichever lands second takes both.

## Checklist

- [x] Every token still goes through the lane rounds. No serial path, and nothing that needs drafts off.
- [x] Drafted output equals `"draft": false`, a resumed prompt equals a fresh one, and concurrent equals solo. (No
      serving path changes.)
- [x] No precision traded for speed. If the bits change, the description says which sums change and why.
- [x] Prompt processing is no slower than the last release.
- [x] The description says which platforms I ran, Metal M1 to M5 and CUDA, and which I could not.
- [x] New tests fail before the change, pass after it, and skip cleanly without their dependency.
- [x] Comments and docstrings are one line. No measurements or history in the source.
- [x] No personal data, machine names, internal hosts or local paths. No AI attribution lines.

## Update 2026-10-13: slices + SM-count blocks (commit dc54a0e5, pushed; description not yet updated)
Cause of BobClawblaw's 5090 -4 % after a 4.7k prompt: serial decode's shared kernel ran 8 blocks (2 groups x 4 key
heads), merge 4 blocks. Change: a group's 4 warp slices write their own partials (no in-block combine), the merge
combines them in the same order (same bits) and runs a block per (row, key head, head); where the four-warp grid
would leave SMs idle (TreeAttention.sms), each slice takes its own one-warp block. GB10, 27B long prompt 128 tokens,
per call: shared 96.6 -> 92.3 (4-warp) / 89.5 us (1-warp), part 32.0 -> 28.4 / 27.3, merge 7.0 -> 4.7; attention total
275.8 -> 255.2 / 247.2 ms (Triton 244.3). Same bits: tf-cuda-test tree (both forms byte-equal), nll 4,092 logit rows
byte-identical to the old build in both forms, draft 4 prompts and multi 8 streams same sha and rounds. Mutation: the
merge dropping slice 3 -> 19x tolerance (dropping it only from the max is benign: same softmax). Fork q27-glue
796099cc carries it (TF_TREE_SMS=0 forces four-warp blocks) for a 5090 rerun.
