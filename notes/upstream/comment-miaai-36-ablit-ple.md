DRAFT — needs the user's go. MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue #36 (@15ky3), 0 comments (2026-09-09).

You are not stupid — there is an open issue on this repo, **#34**, that describes a concrete mechanism
which would produce exactly the symptoms you report, and it has had no reply either.

**Short version:** if you are running `ABLIT=1`, the server may be loading the *stock* model's packed PLE
table underneath the abliterated weights. `ABLIT_META.json` declares `"edit_ple": false`, `start.sh` keys
the packed-table cache on that flag, but @witt3rd measured that 7 shards holding
`ple.ple_embedding.ngram_embedding` genuinely differ from stock. The flag does not describe the bytes.

**Why that gives your symptoms instead of a crash.** The PLE is a very large n-gram embedding table
consulted on every token. Shapes and file lengths match (#34: 17 of 34 shards differ in content, *none* in
byte length), so nothing errors — the model loads clean and then generates from systematically wrong
embeddings. Fluent-but-wrong output, ignoring the instruction, and looping is what that looks like from
the outside.

**A discriminator you can run in a minute.** Prompt-following that is broken by *corruption* is
invariant to sampling; broken by *settings* it is not. If temperature, top-p, and the chat template make
no difference at all, that points at the weights/table rather than at your config. Then check which cache
directory the container actually loaded — `start.sh` names it after the source repo. If you are serving
the ablit checkpoint and the log shows a `ple_cache/Mia-AiLab--…` path, that is #34. The workaround in
that issue is to set `"edit_ple": true` in the local stage's `ABLIT_META.json` so the checkpoint's own
table gets built.

**What we can and cannot contribute.** We do not run the abliterated checkpoint, so this is a hypothesis
about your report, not a diagnosis of it. What we can say from our own measurements on this architecture
(GB10, single Spark, vLLM, the RadixArk NVFP4 build) is that it is *sensitive to PLE row correctness*: we
had a defect where each forward consumed the **previous step's** PLE rows — a far smaller perturbation
than a different table entirely — and it still moved logprobs and top-1 tokens measurably on ordinary
prompts. Serving a whole other model's table is a much larger version of that.

One general rule that has cost us real time here: **verify checksums, not sizes.** We once had a shard
that was the correct length and silently wrong inside; it loaded without complaint and emitted fluent
garbage that was invariant to every config change we tried. Byte-length equality is exactly the case that
looks fine and is not.

*AI assistance was used in preparing this comment; the measurements referred to are ours.*
