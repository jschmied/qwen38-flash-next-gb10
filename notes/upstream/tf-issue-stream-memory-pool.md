# TF issue draft: stream memory as one core pool (not per family)

POSTED 2026-10-13 as https://github.com/ashhart/TensorFold/issues/781 (user: "go"). Origin: "we need a common solution not this per model cooking".

EDITED 2026-10-13 (user: "if its out correct it"): body corrected per tf-781-page-pool-research.md (Pages = interface; no pool grows past admit reservation; point 4 deferred). Live body below.

Title: Proposal: streams take memory from one core pool as they grow, instead of a full window each at load

## What happens now

`--parallel N` reserves N whole `--context` windows at load, and each engine sizes them its own way:

- CUDA host (`native/cuda.zig` -> `cuda_memory.admit`): one rule, but every stream costs the family's
  `stream_bytes` at the full window (Nemotron: `state.Seq.bytes(c, max_len, ...)`), whatever the request needs.
- Metal: Qwen3.8-27B and GLM take `streams` / `streams_fixed` into their own `open` and size their caches
  themselves.
- Paged attention caches exist, but per family: DeepSeek-V4.1's page pool and residency budget (#637), and the
  HIP Qwen3.5 engine's pages (not on `main` yet), which `core/prompt_radix.zig` (#625) holds through its `Pages`
  interface. Both reserve prompt + `max_tokens` at admit and fill pages lazily; neither grows a stream past that.

So a 2k-token chat holds as much memory as a 128k agent session, the slots go to the prompt cache only after
every stream got its worst case, and each new family writes its own answer to "how many streams fit".

## Proposal

One owner for stream memory in core, every family plugged into it:

1. A family declares its stream layout instead of a byte total: fixed bytes per stream (recurrent states,
   Mamba/GDN, drafter state, graph buffers) and bytes per token for each paged cache (attention K/V), plus its
   page size.
2. Core keeps the book of one pool of pages (ids, refcounts, reservations); each engine keeps the device side
   (tensors, page tables). #637's `sessions.Pool` already splits it this way and has most of it (several caches
   per page, rows per token, dirty-range table uploads). It would implement `Pages`, the interface
   `prompt_radix` uses. Lanes admit a request when its fixed part plus the pages for prompt + `max_tokens` fit.
   The prompt cache's kept pages come from the same pool, so evicting cached prompts is the first answer to a
   full pool.
3. The family's attention kernels read K/V through a per-stream page table. Families that already page (#625's
   HIP engine, #637) map onto it; the others change their cache addressing once.
4. `--parallel N` becomes a cap on streams, not a reservation. Growing a stream past its admit reservation, and
   pausing the youngest stream when the pool runs dry, would be new in every engine; a later step, not the first.

## What stays as it is

- Exactness: a stream's bits do not depend on which pages it holds, so drafted == plain and concurrent == solo
  must still hold; the tests that check them now would check this too.
- Captured graphs: a graph keeps fixed addresses for the page table and the state slots, not for the pages.

## Questions before any code

- Is core the right owner, or should this stay per engine (CUDA host, Metal) with a shared interface?
- Is #637's page pool the base to generalise? (`prompt_radix`'s `Pages` is an interface, not a pool; nothing on
  `main` implements it yet.) And which prompt store stays, `prompt_radix` or #637's own?
- Page size: one per pool, or per family?

We can take the CUDA side (Nemotron first, then the shared attention modules in #746) once the shape is agreed.

_Edited: #625's `Pages` is an interface, not a pool; neither paged pool grows a stream past its admit reservation._

