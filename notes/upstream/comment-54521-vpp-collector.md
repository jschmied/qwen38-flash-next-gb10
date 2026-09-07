DRAFT — needs the user's go. GitHub vllm-project/vllm issue #54521, reply to ZC502 (2026-09-07).

@ZC502 I read both files and tried to run the collector on sm_121. Review is clean; the run is
blocked, and the reason is worth reporting back.

**Code**: the only subprocess is `git rev-parse HEAD` (arg list, no shell, 2 s timeout, exceptions
swallowed), no network, no `eval`/`exec`/`pickle`, writes only to `--out`. Nothing I would hesitate to
run.

**Blocker**: `collect_vllm.py:253` constructs `LLM(model=..., **llm_kwargs)` and that is the only
mode. On this box the model under test does not fit that path — a fresh offline engine loads the
checkpoint through `EngineCore` *and* `PleOffloadWorker` concurrently, and on GB10's unified 128 GB
pool the sum does not fit. Three attempts, from 119 GB free after dropping the page cache:
`gpu_memory_utilization` 0.85 → 10 GB free, 0.55 → 0 GB free and 34 GB swapped. The PLE half does not
scale with that knob, so there is no setting that works. The same model serves fine all day as a
long-lived server.

**The suggestion**: add a client mode that POSTs to an OpenAI-compatible endpoint and reads
`prompt_logprobs` from the response, instead of building its own `LLM`. As it stands the tool can only
measure models small enough to construct in-process — and on unified-memory parts those are not the
models with this bug. A server-mode collector would also let you gather from any deployment without
matching its launch flags, which seems closer to what a parity tool wants anyway.

Add that and I will run it against our existing config the same day, on the #55122 cases (1,460 /
1,999 / 5,960-token prompts, no spec, prefix cache off, temperature 0, sequential and concurrent) —
which is the sm_121 collector validation you asked for.
