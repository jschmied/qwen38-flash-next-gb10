POSTED 2026-10-02. TensorFold PR #248 comment (2026-10-02).
The 512 limit reads as a floor in its comment but is written as a cap:

https://github.com/ashhart/TensorFold/blob/1670b61/src/tensorfold/families/qwen4_exp/cuda/multi.py#L237-L238

`if limit > 512: limit = 512` turns every larger share-sized pass into 512 rows. `tests/test_flashnext_prompt_pieces.py::test_both_planner_paths_bound_live_pieces_and_restore_idle_width` catches it: it fails on this branch (`[(…, 0, 512)] == [(…, 0, 960)]`, line 105) and passes on v0.6.2. `limit = max(limit, 512)` would match the comment and keep the 960.

Written with AI assistance (Claude Code).
