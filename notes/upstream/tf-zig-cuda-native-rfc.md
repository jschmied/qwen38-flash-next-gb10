POSTED as https://github.com/ashhart/TensorFold/pull/443 — user go: "do the PR as RFC" (2026-10-06). ashhart/TensorFold draft PR, base zig-flashnext, head jschmied:zig-cuda-native.
TITLE: RFC: zig cuda: serve through tensorfold-native, with a CUDA family registry (Nemotron first)

**RFC, draft.** ZIG-PREVIEW.md lists "the CUDA backend … needs the server wiring and more families". This is a proposal for
that wiring. Before I go further, I'd like to know whether the shape suits you.

## Design question

`native/metal.zig` opens each family inline (`if model_type == …`). For CUDA this PR puts the device, the lane round loop
and the lane host in one generic file, `zig/src/native/cuda.zig`. Each family is one entry in its `registry` and brings
only its lane backend through `open`: `families/nemotron/cuda_native.zig`, about 60 lines, loads the engine, MTP head and
`Lanes` the way `tensorfold lanes` does. Flash Next (NVFP4) or another family would then add an entry and its
`lanes.Backend`, with no server code. Would you rather keep the per-family pattern of `metal.zig`?

## What changes

- `zig/src/native/cuda.zig` (new):
  - `backends`, `families` (built from the registry), and `chip()` (`nvidia-sm<capability>`, e.g. `nvidia-sm121` on a
    GB10);
  - `open()` checks the context, finds the kernel set (`TENSORFOLD_CUDA_KERNELS`, else
    `share/tensorfold/cuda/sm<cc>` beside the binary, as the CLI does), reads `TF_CUDA_DEVICE`, then builds
    `lanes.Config`, `lanes.Engine` and `LaneHost`.
  - The lane host steps rounds on its own thread, and a CUDA context is current per thread. So the family's backend is
    wrapped so that its first call on a thread makes the context current; the family code doesn't see it.
- `zig/src/families/nemotron/cuda_native.zig` (new), and `nemotron.native` exported from `cuda.zig`.
- `zig/build/cuda.zig`: `zig build native` on Linux installs `zig-out/native/bin/tensorfold-native` with these engines.
  The adapter's tests join the host tests.
- `zig/src/core/root.zig`: `core` takes the tokenizer as a module (`@import("tokenizer")`). Otherwise the server's
  `tokenizer` module and `core` would both own `tokenizer.zig`, which Zig refuses. The Metal build doesn't build `core`.
- `zig/src/server/auth.zig`: the key-file checks use `statx` on Linux, because Zig 0.17's `std.c` has no `fstat`/`stat`
  there. Every other OS keeps the existing `stat`/`fstat` path unchanged.

## Tested

- Machine: DGX Spark (GB10, sm_121), Zig 0.17.0, nvcc 13.0, base `zig-flashnext` 88c424e.
- `zig build`, `zig build native` and `zig build test` (the host tests, including the two new ones) all pass.
- `tensorfold-native capabilities --json` on the GB10 reports `"chip": "nvidia-sm121"`, `"backends": ["cuda"]`,
  `"families": {"nemotron_h": ["mlx-q4g64"]}`, and `--backend` takes `auto` and `cuda`.

## Not tested yet

- **Serving Nemotron end to end**, and so all exactness receipts: drafted against `"draft": false`, concurrent against
  solo, and server against `tensorfold run` / `gate.py` token hashes. That needs the kernel capture
  (`zig/tests/cuda/nemotron/box`) and the checkpoint on the box. I'll add the receipts here if the design is welcome.
- No macOS build: the `auth.zig` change keeps the non-Linux path as it was, but I haven't compiled it on a Mac.

## Open points, besides the design question

- The chip class name for gate cells: `nvidia-sm121`, or something else?
- `prefill_step` is the engine's 2,048-row chunk. Graphs stay off, as in `tensorfold lanes`, because lane windows vary.
