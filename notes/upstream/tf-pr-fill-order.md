POSTED 2026-10-01 as https://github.com/ashhart/TensorFold/pull/174 — user's go 2026-10-01 ("post whats ready"). ashhart/TensorFold PR from jschmied/TensorFold:pr-fill-order.

TITLE: CUDA: requests that arrive while a Flash Next prompt fills are admitted between its passes and fill fewest rows first

With `--parallel`, a request that arrives while a long prompt fills and no reply is decoding now gets its first token
in seconds instead of after the long prompt. On one DGX Spark (EXL3 3.05 bpw, `--context 65536 --parallel 8`), four
2k-token requests sent 5–35 s after a 32k-token prompt:

| | first token, requests 1 / 2 / 3 / 4 | the 32k prompt's first token |
|---|---|---|
| main (0.6.0) | 46.5 / 39.3 / 32.1 / 22.4 s | 45.5 s |
| this PR | 6.2 / 7.5 / 6.1 / 7.7 s | 56.7 s |

## Why they waited

Two things, and the first hides the second:

- With no stream decoding, [`_fill`](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/families/qwen4_exp/cuda/multi.py#L227-L235) runs the
  filling prompts' passes back to back until they end and never returns to the scheduler, so a request that arrives
  meanwhile is not admitted until then. (0.6.0's prompt passes inside the decode rounds cover the case with replies
  decoding, the one #162 was about; this is the case with none.)
- Once admitted, prompts take pass rows [oldest first](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/families/qwen4_exp/cuda/multi.py#L254-L266),
  so a short prompt behind long ones waits for all of them. With four ~120k-token prompts at once and 2k-token
  requests every 15 s, the short ones waited 630–715 s for their first token.

## The change

- The scheduler hands the decoder its waiting queue's foreground check (`MultiDecoder.arrived`), and `_fill` returns
  between passes when a request waits, so it is admitted and the next pass can take its rows.
- Passes order the filling prompts by the Mac scheduler's rule ([`_next_fill`](https://github.com/ashhart/TensorFold/blob/c4646171139ee8a3c38103eaa1699dad226ec12b/src/tensorfold/server/prompt_fill.py#L134-L140)): any prompt passed
  over `FILL_GUARD` (8) passes first, then foreground before background, then fewest rows left, oldest among equals.
  The guard bounds a long prompt's wait under a stream of short ones.

Replies do not change: on the branch, `tools/bench_concurrent.py --levels 1,2 --alone --serial` gives every
concurrent reply equal to its request alone (36/36) and every alone run equal to serial (12/12). The 32k prompt's
first token comes later by about the short prompts' passes (+11 s here).

## Tests

`tests/test_flashnext_fill_order.py` (CPU): the order rule (fewest rows left, background last, a prompt passed over
`FILL_GUARD` passes goes first, counts dropped with prompts that left) and `_fill` stopping for a waiting request.
CPU suite: the same failures as main on this machine (missing MLX and similar), none new.
GPU (one GB10): `tests/cuda/test_flashnext_multi.py` 27 passed, as on main.

Not measured: Flash Next on two ranks, other families' concurrent decoders (they don't expose `arrived`, so they keep
today's behaviour), and the four-120k-prompt mix on this branch.

Written with AI assistance (Claude Code); the author reviewed every change.
