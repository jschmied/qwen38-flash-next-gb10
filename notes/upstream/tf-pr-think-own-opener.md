POSTED 2026-10-03 as #327. ## What this changes

A model whose chat template leaves `<think>` to the reply writes the tag itself. Kolibri 1 does this, and so does any template that doesn't open the block in the prompt. The CUDA and Mac servers then sent `reasoning_content` starting with a literal `<think>\n`.

`split_thinking` now strips a leading `<think>` when the markers have no opener. While a reply streams, a tail that may still become the tag is held, the same way a partial `</think>` is held now. A reply without the tag is split as before.

## Receipt

- Environment: TensorFold `main` at 609ca41 plus this commit; PyTorch 2.13.0+cu130; one DGX Spark (GB10). Served `Aleph-Alpha/Kolibri-1` @ e52eb462 through the Kolibri family in the PR stacked on this one.
- Served, before and after: `"reasoning_effort": "low"` on "Was ist 17 mal 23?". Before, `reasoning_content` began with `'<think>\nDer Nutzer fragt…'`. After, it begins `'Der Nutzer fragt…'`, streamed and not, with the same `content`.
- Tests: `tests/test_think_opener.py` (new) failed before the change (1 of 2) and passes after. `tests/test_think_call.py`, `test_lane_stream_text.py`, `test_thinking_off_channels.py`, `test_close_open_think.py`, `test_cuda_thinking_controls.py` and the new file: 116 passed, 6 skipped.
- Not run: a Mac. The change is in `server/text.py`, which both servers use. No decode or prompt timing either, because it touches only the text split after decoding.
