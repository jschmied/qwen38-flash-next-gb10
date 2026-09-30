# Pasture prompt (2026-09-30, user: "at the end do two runs with: [pasture prompt]"), written before the run

Same harness as the scene prompt (`tools/fishscene/gen.py` with `GEN_PROMPT`, seeds 1 and 2, 100k output budget, the
model's default sampling and thinking) on TensorFold + EXL3 3.05 bpw and vLLM (our checkpoint), `check.py` with
`CHECK_LIMIT_KB=30`, then a look in headless Firefox.
- Harder than the fish prompt (4 species with counted features, sun, clouds, trees): expect **0–2 of 4** replies to
  meet every requirement on inspection; leg counts and "recognisable without a label" are the likeliest failures.
- 30 KB limit: 2–4 of 4 under it (the fish replies ran 19–22 KB against 18 KB).
- No engine difference separable from seed-to-seed spread at n=2 per engine.
