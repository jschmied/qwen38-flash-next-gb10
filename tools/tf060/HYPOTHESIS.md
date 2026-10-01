# TensorFold 0.6.0 on the Spark (2026-10-01, user: "do the update and test"), written before the run

0.6.0 (`c464617`) replaces fixed per-slot KV with slots that start at 256 rows and grow in 8,192-row steps behind a
memory gate (2 GiB reserve, kept prompt ends evicted first, new requests wait, the newest live stream stops only when
even the oldest cannot grow), does not hold the mapped n-gram tables back (they page from disk), runs prompts at bf16
by default, fills prompts inside decode rounds, and keeps our block-FP8 path with `lm_head` on FP8G (#126).
Servers: `tensorfold serve <ckpt> --port 8092 --name flashnext --context 262144 --parallel 8`, one start each on
the EXL3 3.05 bpw build and on our `qwen38-flash-next-mtpfp4`. Nothing else on the box.

- **Exactness:** `bench_concurrent --levels 1,2,4 --alone --serial`: every concurrent reply equals its request alone
  and every alone run equals serial, on both checkpoints (as on 0.5.0).
- **Decode, our checkpoint, c=1** (`bench_openai --tokens 400`): `lm_head` on FP8G reads 0.68 instead of 1.27 GB a
  token, so **+3…+10 %** over 0.5.0 + our branch (fibonacci 59.4, chat 37.6 tok/s greedy, TF#126 table).
- **TTFT** (`prefill_cold`, 2k–65k): the single-request path is not what "prompts inside decode rounds" speeds up,
  and bf16 prompts cost more than FP8 ones, so still **2.5–4× vLLM's** (vLLM prod 2.6 s at 8k, 9.4–9.9 s at 30k).
- **Long mix** (`longmix.py`: 4 × ~120k-token prompts at once, 300 tokens each, plus 8 short 2k-token requests every
  15 s): all 12 finish, **0** "ran out of memory" stops; MemAvailable never below the 2 GiB reserve; short requests
  decode at **15–35 tok/s** while the long ones run; the n-gram table's page faults rise during the long prefills
  but no short request decodes below half its idle rate.
- Out of range on any line → read the server log and the probe output before the next step.
