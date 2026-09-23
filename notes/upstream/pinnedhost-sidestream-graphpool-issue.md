POSTED 2026-09-23 as vllm-project/vllm#58441, in the SHORT version (notes/upstream/pinnedhost-sidestream-graphpool-issue-posted.md). This long version was not posted.

# [Bug]: Qwen4Exp PinnedHost PLE prefetch reads n-gram ids from the CUDA-graph pool after release (replay race)

### Your current environment

DGX Spark (GB10, sm_121, unified memory), TP=1, driver 580.178, CUDA 13.0, torch 2.13, vLLM main `1ea7c63f4`
nightly wheel. Qwen3.8-Flash-Next (NVFP4 body, FP8 PLE table), MTP n=3, `cudagraph_mode=PIECEWISE`.

### 🐛 Describe the bug

`Qwen4ExpPLEPinnedHostEmbedding.start_prefetch`
([ngram_embedding.py#L493-L506](https://github.com/vllm-project/vllm/blob/9f07d023d/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L493-L506))
launches the table lookup on a side stream and returns; `_finalize_prefetch` joins it later. The ids the side
stream reads (`gathered_ids`) are produced inside a captured graph segment, so they live in the CUDA-graph
memory pool:

```python
gathered_ids = self._gather_dp_ids(ngram_ids, slot_size)
prefetch_stream.wait_stream(torch.cuda.current_stream())
gathered_ids.record_stream(prefetch_stream)
with torch.cuda.stream(prefetch_stream):
    self._lookup(gathered_ids, output=active_output)
```

Once `start_prefetch` returns, the graph treats that tensor as dead, so later captured segments can reuse its
pool memory. At replay, the main stream can overwrite the ids while the side stream is still gathering with
them. `record_stream` does not protect memory inside a graph pool. The failure is silent: greedy outputs stop
being reproducible.

**What we measured.** We could not run the stock pinned backend on this machine: its pinned 47.7 GiB table does
not fit next to the weights in 128 GB of unified memory. We measured a storage backend that runs **exactly this
start_prefetch/finalize flow** and differs only in where the rows are stored (pageable file mapping instead of
pinned memory; PR TBD). Protocol: 8 greedy prompts sent cold, then the same 8 again warm, within one server
start. Reproducible means identical outputs.

| lookup flow | cold == warm | max \|Δlogprob\| |
|---|---|---|
| side stream, ids from the graph pool (the pinned flow) | 2/8 | 1.41 |
| same + #57785 (sync in `_begin_segment`) | 1/8 | 1.41 |
| side stream, ids first copied to a persistent buffer outside the pool; join still in finalize | **8/8** | **0** |
| lookup on the current stream | **8/8** | **0** |

Only the ids' storage differs between rows 1 and 3, so the pool reuse is the cause. #57785 addresses a
capture-time hazard and does not change this replay-time race. With pinned host memory the lookup is faster than
with page-faulting reads, so the window is narrower; we expect it still exists, but have not shown it on the
pinned backend.

**Possible fixes:**
- copy `gathered_ids` into a persistent (non-pool) buffer on the current stream before the side-stream launch;
  that is row 3 above;
- or run the lookup on the current stream; that is row 4, which gives up the overlap.

A second question worth checking: whether `active_output` / `_prefetch_buffer` (allocated in `__init__`, outside
the pool) is safe for the same reason. In our measurements it was.

Happy to run a patch on GB10 for the backend we can run.

*AI assistance: this report was prepared with Claude; the measurements were run and checked by the submitter.*
