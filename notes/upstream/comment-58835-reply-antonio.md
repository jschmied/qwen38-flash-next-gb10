POSTED 2026-09-28 (user's go: "yes, post to 58835"). vllm-project/vllm#58835, reply to antoniocuegervas's data point.

@antoniocuegervas thank you, that is the case this PR's description undersells: under concurrent agent traffic the faults are not limited to a cold window after start. I'll reword the description around that.

On the one design difference, the 250 ms wait: we measured a gated wait against no wait at one stream, and the no-wait fill was faster on 12 of 12 paired cold requests (−0.14 to −0.60 ms per step). Once the readahead is in flight, the GPU's remaining faults land on reads that are already under way, so the host keeps its one-step lead without waiting ([§4x](https://github.com/jschmied/qwen38-flash-next-gb10/blob/05becde3c22b19bfeaf640d34be792a7e381f6ea/notes/speed-of-light.md#L781-L813)). With eight sessions each step touches more pages and the NVMe queue is deeper, so the balance could differ. Did the 250 ms bound fire often in your runs, or did you compare wait against no wait under that load?

Written with AI assistance (Claude Code).
