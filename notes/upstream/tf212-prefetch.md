POSTED on #212 (2026-10-02, user: "ok, post to 212"): https://github.com/ashhart/TensorFold/pull/212#issuecomment-5949573427

One more for this kernel, if you want it: on GB10, ncu shows `prompt_kernel<2,6,4>` waiting on its own trellis loads
(46 % of warp stalls long_scoreboard, memory throughput 21 %). The next chunk's words are loaded right after this
chunk's decode and needed one barrier and one chunk of mma later. Keeping two chunks in flight (a ring of `wn`,
4 registers more at 3 bits) leaves the decode, the mma and every sum as they are:
https://github.com/jschmied/TensorFold/commit/0809a5c

Flash Next EXL3 3.05 bpw, three alternating rounds against this PR: 8k prefill 5.27–5.30 → 4.53–4.56 s (−14 %), 32k
21.22–21.34 → 18.28–18.31 s (−14 %), the first 16 tokens identical; gate|up alone 7.9–8.1 → 5.2–5.4 ms at 2,048
rows. Three or four chunks were no better, and down slowed at four. test_exl3_prompt_experts and the EXL3 tests pass
(90). [Data](https://github.com/jschmied/qwen38-flash-next-gb10/tree/82e294e855b58aec7eaf324023cb65a018d28472/notes/data/tfexl3x)

Happy to send it as a PR against your branch, or leave it for after this one lands.

Written with AI assistance (Claude Code).
