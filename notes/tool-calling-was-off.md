# Flash-Next could not serve tool calls at all

Found by asking the one question the whole benchmark suite cannot: *does it do tool calls?*

## The defect

`serve-fnext.sh` set `--reasoning-parser qwen3` and nothing else. Any request carrying `tools`
was **rejected outright**:

```
HTTP 400  "auto" tool choice requires --enable-auto-tool-choice and
          --tool-call-parser to be set
```

Not degraded, not mis-parsed — refused. An agent client fails on its first turn. Plain
completions were unaffected, which is why nothing caught it: **throughput, acceptance, NLL,
divergence and coherence tests never send a `tools` field.** A model can benchmark at 36.5 tok/s
and be useless for agent work, and our suite is structurally blind to the difference.

**It was an omission, not a decision.** Both Laguna launchers already set
`--tool-call-parser poolside_v1 --enable-auto-tool-choice` alongside their reasoning parser.
Flash-Next was the only served model missing it.

## The fix

```
--enable-auto-tool-choice --tool-call-parser qwen3_xml
```

Applied to `serve-fnext.sh` and `serve-fnext-prof.sh` (backups `*.pre-toolcall`).

**Picking the name took two attempts.** The reasoning parser is `qwen3`, so `--tool-call-parser
qwen3` looks right and is **INVALID** — the tool side registers only as `qwen3_coder` and
`qwen3_xml`, which are aliases for the same `Qwen3EngineToolParser`. The class was right, the
name was not. Pre-flight instead of inferring:

```python
import vllm.tool_parsers as tp, inspect, re
names = set(re.findall(r'^\s{4}"([a-z0-9_\-]+)":', inspect.getsource(tp), re.M))
```

Note also that this build moved to `vllm.parser.engine`, so the old `ToolParserManager` registry
reads **empty** at import — enumerating it the obvious way returns nothing.

**Why `qwen3_xml` and not `hermes`:** the chat template emits XML-nested markup, not JSON in
tags —

```
<tool_call>
<function=name>
<parameter=key>value</parameter>
</function>
</tool_call>
```

`Qwen3Parser`'s own docstring shows exactly that shape (`TOOL_CALL_START`, `FUNC_PREFIX =
"<function="`, `PARAM_START = "<parameter="`). `hermes` would have mis-parsed silently.

## Result: 32/32

`fp8head`, MTP k=2, 8 attempts per temperature, counting whether a parsed `tool_call` returns:

| temperature | tool_call | finish_reason | function |
| --- | --- | --- | --- |
| 0.2 | **8/8** | `tool_calls` x8 | `run` |
| 0.6 | **8/8** | `tool_calls` x8 | `run` |
| 1.0 | **8/8** | `tool_calls` x8 | `run` |
| server default | **8/8** | `tool_calls` x8 | `run` |

Correct function name every time. **Flash-Next patterns with qwen38 RadixArk (8/8 at every
temperature), not with qwen36-35b/A4Q where temp 1.0 breaks tool calling** — so the per-model
re-test in `[[qwen-nvfp4-agentic-temp]]` was worth running, and our 0.6 default is not required
by tool calling on this model.

## The rule

**A serving config has capabilities, not just speed.** Every number we produced for this model
was valid and none of them touched the capability that decides whether it can be used at all. Any
new serve recipe gets a tool-call probe before it gets a benchmark — one request with a `tools`
field, checking for a parsed `tool_calls` and the right function name.
