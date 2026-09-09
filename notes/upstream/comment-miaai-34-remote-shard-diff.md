DRAFT — needs the user's go. MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark issue #34 (@witt3rd), 0 comments (2026-09-09).

Two things to add to this: an independent check of one of your numbers, and a way for anyone to redo the
whole comparison without downloading either checkpoint.

**The denominator checks out.** `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` currently has exactly **34** numbered
shards (`model-00001-of-00034` … `model-00034-of-00034`) plus `amax.safetensors`,
`amax_checkpoint.safetensors` and `amax_checkpoint.json`. So the CHANGELOG's "9 of 37" is off in both
numerator and denominator, and your "17 of 34, 35 counting `amax_checkpoint`" matches the repository as
published.

**The diff does not need a download.** The HF API exposes the per-file sha256 as `lfs.oid`, so the
shard-level content comparison is two HTTP calls:

```
GET https://huggingface.co/api/models/<repo>/tree/main?recursive=1&revision=<sha>
```

Each LFS entry carries `{"oid": "<sha256>", "size": …}`, and `oid` is the same hash `sha256sum` gives for
the downloaded file. Diffing the two snapshots' `path → oid` maps reproduces your table remotely — useful
here because the ablit repo is gated (the request still needs the caller's token, but no bytes move), and
useful in general for checking a *published* artifact against its own metadata before anyone pulls
120 GiB. We built this into our provenance tooling after finding that only 4 of 30 checkpoints we were
archiving had any recorded origin at all.

**On "why it matters", from our side.** We serve the RadixArk NVFP4 build on a single GB10 under vLLM,
not the ablit checkpoint, so we cannot confirm your case directly. But we can confirm the sensitivity your
argument rests on: we had a defect in which every forward consumed the **previous step's** PLE rows —
strictly smaller than substituting a different model's table — and it still changed logprobs and top-1
tokens on ordinary prompts, and was invisible to a "same prompt twice" check because two identical
consecutive requests agreed with each other. Mixed ablit body weights over stock PLE is the larger version
of the same error, so we think your "the served model is then a mix" reading is the right one to act on.

**Possibly the same bug, reported from the user end: #36** ("doesn't follow any prompt, speaks bullshit
and repeating itself"), also unanswered. Symptom-wise that is what we would expect from serving the wrong
table, and it is worth checking whether that reporter is on `ABLIT=1`.

The `edit_ple` flag being a *declaration* rather than a *measurement* looks like the general fix here:
deriving it from the shard hashes at stage time would have caught this without anyone hashing 120 GiB by
hand.

*AI assistance was used in preparing this comment; the shard count was checked against the live HF API and
the PLE-sensitivity measurement is ours.*
