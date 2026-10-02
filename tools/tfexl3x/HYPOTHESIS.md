# #212's prompt kernel: trellis-word prefetch depth (2026-10-02)

ncu on `prompt_kernel<2,6,4>`: 46 % of stalls long_scoreboard (global loads), memory 21 % → the next chunk's words,
loaded one chunk ahead, arrive late. Ring of PREFETCH chunks (same bits). Microbench (`pbench.py`, R 2,048):
gate|up 7.9–8.1 → 5.2–5.4 ms (−34 %) at PREFETCH 2; routed() −21.5 %; hashes identical.
Prediction, real model (Flash Next EXL3 3.05, `tfexl3/prefill_ab.py`, 3 alternating rounds vs #212):
- prefill −8…−13 % at 8k and 32k (gate|up is 32.5 % of an 8k prefill on #212, down 10.3 %).
- the first 16 tokens identical; decode untouched (prompt kernel only past 64 rows).
- **Result (§5bj):** −13.7…−14.3 % at 8k and 32k, every round, tokens identical (slightly above range).
