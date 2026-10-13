# TF issue draft: stream memory as one core pool (not per family)

POSTED 2026-10-13 as https://github.com/ashhart/TensorFold/issues/781 (user: "go"). Origin: "we need a common solution not this per model cooking".

Title: Proposal: streams take memory from one core pool as they grow, instead of a full window each at load

## What happens now

`--parallel N` reserves N whole `--context` windows at load, and each engine sizes them its own way:

- CUDA host (`native/cuda.zig` -> `cuda_memory.admit`): one rule, but every stream costs the family's
  `stream_bytes` at the full window (Nemotron: `state.Seq.bytes(c, max_len, ...)`), whatever the request needs.
- Metal: Qwen3.8-27B and GLM take `streams` / `streams_fixed` into their own `open` and size their caches
  themselves.
- Paged attention caches exist, but per family: the HIP Qwen3.5 engine's pages with `core/prompt_radix.zig`
  (#625), and DeepSeek-V4.1's page pool and residency budget (#637).

So a 2k-token chat holds as much memory as a 128k agent session, the slots go to the prompt cache only after
every stream got its worst case, and each new family writes its own answer to "how many streams fit".

## Proposal

One owner for stream memory in core, every family plugged into it:

1. A family declares its stream layout instead of a byte total: fixed bytes per stream (recurrent states,
   Mamba/GDN, drafter state, graph buffers) and bytes per token for each paged cache (attention K/V), plus its
   page size.
2. Core keeps one device pool of pages (the `Pages` type #625 already added to `prompt_cache.zig`). Lanes admit
   a request when its fixed part plus the pages for its prompt fit, and give a stream more pages as it grows.
   The prompt cache's kept pages come from the same pool, so evicting cached prompts is the first answer to a
   full pool.
3. The family's attention kernels read K/V through a per-stream page table. Families that already page (#625's
   HIP engine, #637) map onto it; the others change their cache addressing once.
4. When the pool runs dry mid-reply: evict prompt-cache pages first, then pause the youngest stream (its state
   stays, it waits for pages) rather than fail it. `--parallel N` becomes a cap on streams, not a reservation.

## What stays as it is

- Exactness: a stream's bits do not depend on which pages it holds, so drafted == plain and concurrent == solo
  must still hold; the tests that check them now would check this too.
- Captured graphs: a graph keeps fixed addresses for the page table and the state slots, not for the pages.

## Questions before any code

- Is core the right owner, or should this stay per engine (CUDA host, Metal) with a shared interface?
- Is #637's page pool the base to generalise, or `prompt_radix`'s `Pages`?
- Page size: one per pool, or per family?

We can take the CUDA side (Nemotron first, then the shared attention modules in #746) once the shape is agreed.
