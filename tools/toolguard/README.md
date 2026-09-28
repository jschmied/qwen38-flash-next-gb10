# Qwen tool-call parser guards (from blazux/qwen3.8-Flash-DGX, Apache-2.0)

The four `.patch` files are blazux's patches 12 and 13 with their tests, unchanged
(https://github.com/blazux/qwen3.8-Flash-DGX, `src/patches/`). They fix vLLM's `qwen3` tool parser (used by
`--tool-call-parser qwen3_xml` / `qwen3_coder`):
- a literal or malformed `<tool_call>` in reasoning or text no longer switches the parser into a tool preamble
  that discards the rest of the answer;
- inside a fenced code block, `<tool_call>` / `<function=` stay text, and a bare `<function=` header opens a call
  only at the start of a line. Before, writing *about* the tool format produced a phantom call or `content: null`.

**On our base (vLLM main 1ea7c63f4 + our overlay), 2026-09-28:** both apply with `--fuzz=0`. The parser suite
(`tests/parser/engine`, CPU): unpatched 4326 passed / 45 failed (the new cases), patched 4370 passed / 1 failed.
The one failure, `test_parser_engine.py::TestTruncatedToolOpenerStreamParity::test_content_around_unpromoted_tool_block_stays_ordered`,
passes on the unpatched base and encodes the old behaviour (`A <tool_call>garbage</tool_call> B` → `A  B`, the
block dropped); patched it keeps the block as text, which is the fix's intent. Streaming and non-streaming agree.
`install.sh.txt` installs into the prod venv with `*.orig-toolguard` backups.
