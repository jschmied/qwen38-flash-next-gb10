# #593 reply draft: GLM-5.3-Flash EXL3 2.05 bpw at TP=1 on one GB10 (NOT posted)

@BHCC2025, thanks. Receipt for `glm-exl3-wip` pinned at `5da1e007`, at TP=1 on one DGX Spark (GB10, sm_121, CUDA 13), turboderp's `GLM-5.3-Flash-exl3` at 2.05 bpw (79.4 GiB, every shard's sha256 checked):

- `glm-shared-check <ckpt> dc.json 0 1 id`: PASS. 108 rows of 5 streams over 12 rounds (2 to 5 streams a forward, 1 to 4 rows each, greedy and sampled) equal their lone twins' logits, tokens and drafts byte for byte, and every state its twin's.
- `glm-lanes-concurrent <ckpt> dc.json 0 1 id`: PASS, 38/38: all at once, staggered, in reverse, the same prompt twice, prompts a chunk at a time while the others decode, sampled; 540 forwards held several streams' windows.
- Prompts: three chat prompts of 2,998, 2,713 and 2,483 tokens (a Zig source file, a markdown guide, a README), so prompts run in several chunks; capacity 8,192.
- Memory: the lowest MemAvailable over both runs was 23 GiB of the Spark's 128 GB.

One change was needed: the 2.05 bpw quant keeps the MTP layer (`layers.45.*`) in shard 11, not in a separate `mtp.safetensors` as 3.05 and 4.05 do, so `cuda_weights.zig:535` stopped with `FileNotFound`. Treating a missing file as "the MTP layer is in the shards" was enough:

```zig
if (with_mtp and c.mtp > 0) ck.add(dir, "mtp.safetensors") catch |e| if (e != error.FileNotFound) return e;
```

Not measured: speed (the checks don't time decode), long contexts, the HTTP server.
