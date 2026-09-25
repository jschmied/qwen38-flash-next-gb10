DRAFT — needs the user's go. vllm-project/vllm PR #58439, comment after the rebase force-push, replying to hclsys (2026-09-25 11:45 CEST).

<!-- Post only after the force-push of pr/ple-checkpoint-mapped-rebased; replace <HEAD> with the pushed head
     (currently 86f103430), and after the body edit (58439-body-update-0925.md), since the last line says it is done. Thread re-read 2026-09-25 ~11:40 CEST: last comment hclsys 03:29 UTC, nothing newer.
     What the thread lacks: that main moved again after hclsys's rebase, and what the force-push changed. -->

Rebased onto main `378504a54` and force-pushed (head `<HEAD>`). Thanks @hclsys for the three GB10 runs, the
attribute check and the trial rebase.

Main moved again after your rebase, so there was more to it than the `engram.py` add/add:
- **#57497** moved the device/pinned backends to `common/ngram_embedding.py`. The pageable backend stays in
  `nvidia/`. The post-load bind hook and `_join_prefetch_stream()` moved with the base classes.
- **#58086** made `DefaultModelLoader._prepare_weights` return four values. The file resolution unpacked three,
  and the unit tests stubbed it, so they stayed green; mypy caught it. Fixed, and a new test calls the real
  loader (open design question 1).
- **New commit:** `checkpoint_mapped` is rejected off CUDA. Since #57497 the ROCm path would ignore it and
  allocate the pinned table.
- #58489's persistent `_prefetch_ids` is kept; this backend uses it too.
- DCO: the second commit's author email now matches its sign-off.

`test_ple_pageable.py` on GB10: 28/28. `models_basic` Qwen4Exp set on CPU: 53 passed, 42 skipped. pre-commit
clean. The PR body is updated to match.
