# vllm#58449 on GB10 — hypothesis, written 2026-09-24 before any run

**What the PR changes for us.** Prod logs `Fused multi-step draft decode is not supported by attention backend(s)
QWEN4_EXP_EXP_QSA_STATE`. So with MTP n=3 the V2 speculator rebuilds the drafter's attention metadata (host-side
`_build_uniform_attn_metadata`) before each draft decode step. With the PR, the QSA builder updates in place and
the fused loop runs. Draft decodes run eager under our PIECEWISE mode either way ("PIECEWISE cudagraphs are not
supported for draft decodes"), so the saving is one metadata build plus its H2D traffic per verify cycle.

**Arms.** `base` is prod (vllm-venv-main1ea7, mtpfp4, MTP n=3, nodrop, 32k slice, checkpoint-mapped PLE). `pr` is
the same with the PR's `qsa_cache.py` (head 848dbf89e) swapped in. 3 interleaved starts per arm.

**Expected.** An outcome outside these ranges means I debug the instrument first.

| cell | expected |
|---|---|
| fallback log line | present in base, absent in pr (void check) |
| PR unit test on GB10 (`-k draft_decode_metadata_update`) | passes |
| greedy output hashes pr vs base | identical: the target verifies every draft, so metadata only affects drafting, and draft metadata equality is what the PR's test asserts |
| acceptance (rate, mean len) | identical to 3 digits, same reason |
| decode ms/tok, c=1 (streamed, first→last token) | −0 … −3 % |
| decode aggregate tok/s, c=4 | +0 … +4 % |
| agent-loop s/turn | −0 … −3 % |
| TTFT | unchanged ±2 % (prefill does not take the draft decode path) |

**Why small.** One metadata build is sub-ms to ~1 ms of host work per verify cycle, against a ~100+ ms cycle at
c=1. Async scheduling can hide host work, so a null is plausible. A gain above 5 % would mean the rebuild
contained a sync or much more work than assumed, which would need a profile before I believed it.
