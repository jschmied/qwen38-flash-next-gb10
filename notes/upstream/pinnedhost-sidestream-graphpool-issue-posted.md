### Your current environment

DGX Spark (GB10, sm_121), TP=1, CUDA 13.0, torch 2.13, vLLM main (`1ea7c63f4` nightly wheel), Qwen3.8-Flash-Next,
MTP n=3, `cudagraph_mode=PIECEWISE`.

### 🐛 Describe the bug

`Qwen4ExpPLEPinnedHostEmbedding.start_prefetch`
([ngram_embedding.py#L493-L506](https://github.com/vllm-project/vllm/blob/9f07d023d/vllm/models/qwen4_exp/nvidia/ngram_embedding.py#L493-L506))
launches the lookup on a side stream and returns; `_finalize_prefetch` joins later. The `gathered_ids` it reads come
from a captured graph segment, so they live in the graph pool. Once `start_prefetch` returns, later segments can reuse
that memory, and at replay the main stream may overwrite the ids while the side stream is still reading them
(`record_stream` does not cover graph-pool memory). The effect is silent: greedy outputs stop being reproducible.

Measured with a backend that runs exactly this flow and differs only in row storage (a pageable file mapping, #58439).
Protocol: 8 greedy prompts, cold vs warm, within one server start.

| lookup | cold == warm | max \|Δlogprob\| |
|---|---|---|
| side stream, ids from the graph pool (this flow) | 2/8 | 1.41 |
| same + #57785 | 1/8 | 1.41 |
| side stream, ids copied to a persistent buffer first | **8/8** | 0 |
| current stream | **8/8** | 0 |

Caveat: PinnedHost itself could not run here, because its pinned table does not fit in 128 GB of unified memory.
Pinned reads are faster, so the window should be narrower; not reproduced on PinnedHost itself.

Possible fix: copy `gathered_ids` into a persistent (non-pool) buffer on the current stream before the side-stream
launch (row 3), or look up on the current stream (row 4, which loses the overlap).

*AI assistance: prepared with Claude; measurements run and checked by the submitter.*
