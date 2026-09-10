# The calibration corpus

> **⚠️ CORRECTED 2026-09-10: this corpus is 99.94 % ASCII and contains no Thai.** It is described
> below as spanning "roughly ten languages" — those are **programming** languages. SWE-bench
> *Multilingual* is multilingual in code and monolingual in human script. Building on it produced a
> checkpoint that corrupts Thai combining marks; see `combining-mark-regression.md`. Read every
> "diversity" claim below with that in mind.

Material for the Local-Hessian NVFP4 rebuild, and the disjoint pool the eval draws from.

## What we have (2026-09-10)

The corpus needed **no new agent runs**. The trajectories already existed on the x86 SWE box
(`10.0.0.8:/root/gen-multilingual-100/`) from an earlier SWE-bench Multilingual run.

| | value |
| --- | --- |
| instances | 100 (of SWE-bench Multilingual's 300) |
| distinct orgs | 39 |
| languages | Java, JS/TS, Go, Rust, Ruby, PHP, C/C++, Python, Lua, jq |
| tokens (Flash-Next tokenizer + its own chat template) | **1,814,667** |
| per instance | min 3,507 · median 14,089 · max 95,239 |
| over 32k ctx | 12 instances (1 over 64k) |
| chars by role | tool 78 % · user 11 % · exit 6 % · assistant 5 % · system 0.2 % |

Built by `10.0.0.8:/root/mkcalib.py` → `/opt/llm/calib/calib-multilingual-100.jsonl`
(6.3 MB, one object per instance, messages verbatim; the chat template is applied on the
GB10 side by the model's own tokenizer, never baked into the file).
Token census: `/opt/llm/calib/token-census.json`.

## Why this replaces the old corpus

The corpus used for the gate experiments was **150,750 tokens from 3 django instances, Python
only**. This one is 12× the volume and, more importantly, spreads over ten languages and 39
projects. The gap that mattered was never volume — finding: the 21 zero-row experts came from
an 8,192-row capture cap, not from the corpus being small — it was **diversity**.

Routing arithmetic at this size: 1,814,667 tokens × top_k 10 / 512 experts = **35,442 rows per
(layer, expert) on average**. Even an expert taking a twentieth of its share clears a 4,096-row
cap. Volume is no longer the binding constraint; a real routing census still has to confirm the
tail (queued, see `build-driver-design.md`).

The tool role dominating at 78 % is not a defect to reweight away: that is the token mix the
model actually processes in an agent turn, and the routing statistics we want are the ones from
real turns.

## Provenance caveat

**The trajectories are qwen38-27b's; the tokenizer is Flash-Next's own.** Two different things,
easy to conflate. The census loaded `/opt/llm/models/qwen38-flash-next-nvfp4`, whose
`tokenizer.json`, `tokenizer_config.json`, `vocab.json` and `merges.txt` are **bit-identical to
Qwen's official Flash-Next source** on PBS. The two models share a vocabulary (`len(tok)` 248,077,
config `vocab_size` 248,320 for both) but **not a chat template** — 8,952 chars against the 27B's
14,093 — so using the wrong one would have silently shifted every token count through the
template while raising nothing.

The trajectories themselves are 27B's, not Flash-Next's. The tool output, file dumps, test
runs and tracebacks — the 78 % — are the repos' own text and model-independent. The 5 %
assistant text is another model's prose style. Not worth re-generating for a calibration set.

The four `gen-javajs-*` dirs (27b, qwen38, glm47, hy3) add **no new instances** — all 28 are a
subset of the 100 — only four models' assistant text on the same material.

## The disjoint eval pool

Calibrating and then scoring on the same instances leaks. Split by construction rather than by
carving up the 100:

- **calibration** = the 100 instances above
- **evaluation** = drawn from the **200 held-out** Multilingual instances (`/root/m_pool_remaining.json`)

Cost of that disjointness, measured: **1 of the 200 has a cached Docker image**. The 131 images
on the box (180.7 GB) are the 100 calibration instances plus 31 from Lite/Verified work. A
held-out eval therefore pays ~1.4 GB of image pull per instance, and 82 GB of disk is free —
room for ~55 images, so a 100-instance run needs pull→run→rmi churn.

Per `swebench-docker-cache-confound`: **pre-pull every image and verify it exists before the
agent run starts.** A pull timeout inside the run is scored as an empty patch and silently
becomes a quality result.

## Trajectories on the GB10 itself

`/opt/llm/swebench-runs/` holds 32 instances (a4q-nvf4-full 25, uni-4x4 4, a4q-nvf4 2,
a4q-nvf4-ext 1) — django and astropy, i.e. exactly the narrow Python material this corpus
replaces. Kept for reference, not corpus material.

## Statistical limit of run A

SWE-bench resolution rate has SE ≈ 2.9 points at 300 instances, worse on a smaller slice. The
1–2 point effect JasonW2025 reports for W4A16-vs-W4A4 is inside that noise. **Run A is a
regression check — "the rebuild did not break the model" — not the proof that Local-Hessian
helped.** The proof stays the BF16 logprob-divergence measurement.

## v2 — script coverage added, 2026-09-10

The v1 corpus was 99.94 % ASCII with **zero Thai**, and Local-Hessian spent Thai's precision on the
directions that had energy in it (`combining-mark-regression.md`). Fixed at the source.

**Added:** 121 Wikipedia articles across **13 languages / 9 scripts** — Thai, Devanagari, Arabic,
Hebrew, CJK, Kana, Hangul, Cyrillic, Greek, plus Latin-with-diacritics (Vietnamese, German, French,
Spanish, Turkish). 539,348 characters, ~135k tokens.

Fetched through **HF's `datasets-server` rows API**, not a dataset download: `openlanguagedata/flores_plus`
is gated behind an agreement, and Wikipedia's own API returned 429 to a first attempt that polled too
fast. The rows API serves content directly with no archive and no rate-limit fight.

| corpus | ASCII | Thai | combining marks |
| --- | --- | --- | --- |
| v1 | 99.9428 % | **0** | ~0 |
| **v2** | 95.3483 % | **0.5753 %** | **20,685** |

### The share that matters is in the CAPTURE, not the file

At their natural 0.6 % of characters the scripts would still carry almost no energy in H. What builds
the Hessian is the captured rows, so the sampler now **reserves** a share for them
(`SCRIPT_SHARE = 0.30`) instead of letting the length bands compete them away.

Taking *every* script document lands at **53 %** of the capture — non-ASCII tokenizes at ~2.4
chars/token against the agent text's ~4, so they punch well above their character share. That would
over-fit the other way, away from the workload we actually serve. Capped at 30 % and taken
round-robin by language so the cap never drops a script entirely.

Resulting capture: **67 documents, 393,689 tokens — 30 % script over 13 languages, 70 % agent**, with
position coverage still reaching 95,239.

### The lesson, stated plainly

Every stratification I did on v1 — position, project, expert row counts — measured something real
and none of them measured the axis that broke the build. The dataset was called *Multilingual* and I
took that to mean human languages; it means programming languages. **Count the codepoints.**
