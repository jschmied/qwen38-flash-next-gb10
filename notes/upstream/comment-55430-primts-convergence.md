Thanks — checked both.

The kernel does not cross: flashinfer#4996 states "Source supports SM100/SM103", and this PR targets sm_121 (GB10). #56240 treats them as separate work for the same reason.

The interface already matches, though. Their `indexer_block_ids` — logical selected-block IDs `[total_q, block_topk]`, consumed without expansion into token indices — is what this PR calls `block_indices_out` (`[num_tokens, block_topk]`, `-1`-padded, with `expand=False` skipping the expansion). Two independent implementations, same contract.

That suggests a better shape than what I have now: #56240 adds `VLLM_QSA_ATTENTION_BACKEND=auto|triton|prims_ts`, and this PR adds its own `VLLM_QSA_TILE_UNION`, so both landing would give two selectors for one decision. I would rather register the SM121 tile-union path as a third backend under theirs — `auto` selecting prims_ts on SM100/103, tile-union on sm_121, triton otherwise — with the compressed-block-id output as one shared indexer contract. That makes this PR depend on #56240 landing first, which seems the right ordering.

Separately, and relevant to anyone reading the numbers here: the tile-union path was broken by #55272's `projected_qk` refactor (two dangling `hidden_states` references, no try/except, so entering the branch raised NameError). Fixed in `c5d7eba3`. The 1.45x figure predates that refactor and I am treating it as unverified until re-measured.

*AI assistance was used in preparing this comment.*
