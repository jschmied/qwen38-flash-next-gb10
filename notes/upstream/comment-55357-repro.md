DRAFT — needs the user's go. Comment on vllm#55357 (tgmerritt, MTP episodic 0 % acceptance + degenerate output). Numbers: findings 126/127 (`notes/data/acceptcell2*`, `acceptcell3*`); bisect (finding 128) to be appended before posting.

We have a reproducer for this on a GB10 (DGX Spark, sm_121, TP1), and it narrows the trigger: **the corruption starts when two or more prompts are prefilled in the same scheduler step with MTP enabled.** Same model family (Qwen3.8-Flash-Next, FP8-mixed checkpoint), nightly `0.28.1rc1.dev352` and `dev401`, `--speculative-config {"method":"mtp","num_speculative_tokens":3}`, prefix caching on **and off**.

**Reproducer.** Send c identical-length chat prompts (549 tokens, differing only in a salt) at the same instant, `temperature 0`, `max_tokens 128`, `ignore_eos`. In roughly half of the batches, every request but one comes back as garbage from its **second token** on (`"Based" → "Basedsumsumsumsumsum…"` then mixed-script noise), and its draft acceptance sits at ~9 % (per-position 0.09 / 0.06 / 0.05) for the rest of the request; the surviving request is fluent with normal acceptance. Which request survives varies.

| build | config | batches with ≥ 1 corrupted request | corrupted per bad batch |
|---|---|---|---|
| dev401, MTP n=3, prefix cache on | c = 1 / 2 / 3 / 4 / 5 / 8, 10 batches each | 0/10, 3/10, 4/10, 4/10, 4/10, 5/10 | 1 / 2 / 3 / 3 / 3 |
| dev401, MTP n=3, **prefix cache off** | c=4 ×3 | 1/3 | 3 |
| dev352, MTP n=3, cache on | c=2 ×10, c=3 ×6 | 5/10, 2/6 | 1, 2 |
| dev352, MTP n=3, pairs with the second request delayed 0 / 0.15 / 0.4 / 1 / 3 s | 6 each | **3/6, 0/6, 0/6, 0/6, 0/6** | 1 |
| dev352, **no speculation** | c=2 ×10, c=3 ×6, pairs at all delays | **0/46** | — |

So: not the prefix cache (fails with it off), not preemption (counter stays 0, nothing in the log), not prompt-length padding (all prompts 549 tokens), not context length (549 tokens), and not the drafter alone — the target's own output is wrong, the acceptance collapse is downstream of that. A request whose prefill runs next to another request's *decode* (the 150 ms stagger) is fine; only a multi-prefill step triggers it. That also explains "episodic": it needs two prompts to land in one step, which real traffic does now and then and a sequential probe never does.

The two things we have not separated yet: MTP n=1 vs n=3 (multi-step draft decode), and `--enforce-eager`. Running those now and will add the result here. Probe script and raw outputs are in our record repo (link) if useful; happy to run a specific configuration on this box.
