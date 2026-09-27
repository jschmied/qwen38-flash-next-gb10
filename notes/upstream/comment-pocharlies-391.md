POSTED 2026-09-27 (user's go: "ask"). pocharlies-org/k8s-gitops-pocharlies PR #391 (rollback of phase 3b = #58439 + be7a84fe).

Hi @pocharlies, I'm the author of [vllm#58439](https://github.com/vllm-project/vllm/pull/58439) (checkpoint-mapped PLE). I saw phase 3b (#390, with [be7a84fe](https://github.com/jschmied/vllm/commit/be7a84fe46ef143a898fb4a02097c45cfbee57a6)) was rolled back by this PR the same evening. Could you share what failed? Any of these would help:

- the first error or traceback from the head or worker log, or the symptom (hang, OOM, start-up memory gate, or it started but misbehaved);
- the vLLM base (nightly `2a02f6ef`?) and whether you ported only be7a84fe or the whole #58439 branch at that point;
- with TP=2 over two nodes: did both ranks map the table, and was the page cache dropped before the start?

We have only validated TP=1 on one GB10 (hclsys confirmed it on a second single Spark), so a two-node failure is new information. One thing we know on unified memory: the mapped table sits in the page cache, so a free-memory check can count it as used. The note here about the worker's 0.77 gate sounds like that, but I'd rather not guess. If it is a bug in vllm#58439, I'll fix it there.

Written with AI assistance (Claude Code).
