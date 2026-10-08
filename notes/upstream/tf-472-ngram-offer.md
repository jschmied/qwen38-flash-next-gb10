@AdrianBinDC we've run Flash Next on a single GB10 in vLLM since August, and the n-gram table was the hardest part there. If it helps, we can take the GB10 side of layer 2's table against your branch once it's pushed. We won't start anything separate.

What we measured in vLLM, on the checkpoint we serve there (its n-gram table is 47.7 GiB):

- **A pinned host copy doesn't fit.** Body 69.4 GiB + pinned table 47.7 GiB + the loader's page cache went past the 121 GiB pool and hung the box.
- **Reading it in place from the mapped checkpoint works.** That's vllm-project/vllm#58439, which we have run in production since 2026-09-23. It needs no copy of the table, and swap went from about 50 GiB to about 6.
- **How much of it stays resident depends on the KV budget.** With vLLM's default KV, the table was barely resident: about 30 major faults per decode step, and about 60 ms/step against 53.7 ms with a smaller KV cache. Readahead for a step's cold pages is vllm-project/vllm#58835.

We can also run your branch on our GB10 for receipts: Python vs Zig equality, memory, and faults per step.
