POSTED 2026-10-11. ashhart/TensorFold PR #654 comment (2026-10-11).

One more piece I'd like to add here before review, if you agree. The sampler's rule (1/T, top_p, the near window of 20, ln(min_p), top_k, the seed) is currently built in three places:
- `cuda/sample.zig`'s `rule()`, from this PR;
- a copy in `tests/cuda/sample_tests.zig`;
- Metal's `Forward.sample` in `families/nemotron/forward.zig`, which packs the same values as four floats plus two seed halves.

Proposal: `rule(sampling)` lives once in `core/lanes/gpu_rule.zig`, next to the host reference it already matches. Then:
- the CUDA `Rule` struct and the Metal cfg are both filled from it;
- the test drops its copy;
- a host test pins the layout of both.

The kernels and their arguments stay unchanged, and the receipts would be the same as this PR's: `sample` against the host references, plus Nemotron reply hashes on both backends (a Mac is available here for the Metal side).

If you'd rather keep this PR CUDA-only, I'll send it as a follow-up after this one lands.
