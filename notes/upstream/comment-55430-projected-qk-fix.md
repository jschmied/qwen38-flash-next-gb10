Thank you for the measurements, and apologies for the slow reply.

Your data led me to a bug in the PR, and I think it means the union path never executed on your run.

vLLM #55272 ("Remove torch.compile for NVIDIA implementation") split `forward(hidden_states)` into a caller that projects and `_run_qsa(projected_qk, ...)`. Two references in the tile-union path still named `hidden_states`, which is no longer in scope in either function:

- `qsa.py::_run_qsa` — `*self._tile_union_workspace_shape, hidden_states.device`
- `indexer_qsa.py::forward` — `block_indices_out.shape != (hidden_states.shape[0], block_topk)`

`hidden_states` still exists in `qsa.py`, but in the outer `forward`, so this merged cleanly and was silently broken. Neither file has a `try`/`except`, so entering the tile-union branch raises `NameError` before any work is done. Your runs completed and passed the correctness gates, so the branch cannot have been entered — most likely `self._tile_union_workspace_shape` is `None` in the TP=2 configuration. The log lines you quoted are emitted at configuration and warmup time, not per call, so they do not indicate that the path ran.

That would explain the table: the two arms were both the base path, and the differences are run-to-run noise, including the 295K rows straddling the base.

I have rebased onto `main` (126 commits) and fixed both references to use `projected_qk`. Head is now `c5d7eba3`.

If you are still willing to run this, a repeat on the fixed head would be valuable — the TP=2 shape is one I cannot test. Two things worth capturing: whether `_tile_union_workspace_shape` is non-`None` on your ranks, and whether the eligibility check passes with prefill chunked at 8192. Your point about NVFP4 reducing the sparse-attention share of prefill time stands regardless and may well dominate the result.

I would hold off on the tile-config sweep until the path is confirmed active; sweeping configurations for a branch that does not execute would only produce more noise.
