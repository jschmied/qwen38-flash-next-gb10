DRAFT — goes upstream as docs/recipes/training-a-drafter.md only after the RFC (tf-rfc-generic-drafter.md) has an answer.

# Training a drafter

This is how a learned drafter is trained for a target the engine serves, and taken to the speed it gives. The steps
are in order. Each one is a measured win, so a new target goes through them without measuring each again. It comes
from one bring-up, so treat the numbers as Kolibri's, not as laws:

- Kolibri-1 (78B MoE, 3.5B active, FP8) on one DGX Spark, 3-10 Oct 2026. Decode speed against copy drafting alone:
  1.28-1.38x greedy and 1.23-1.47x sampled at one stream, 1.10x at four streams. Released as
  [josch15366/Kolibri-1-block-drafter](https://huggingface.co/josch15366/Kolibri-1-block-drafter).

## How to use it

1. Do the steps in order. Score after each on the same fixed slices, and keep their prompts out of every training set.
2. Score on output the target generated for prompts no checkpoint trained on: per class (code/agent, chat, each
   language), accepted drafts per round of 3 and 4, and agreement per position.
3. Before any speed claim, run the engine end to end against copy drafting alone, the same build, greedy and sampled,
   one stream and several, with the reply hashes equal to the undrafted run.
4. When a step proves out on another target, add its result. When one loses, add it to "Don't bother".

## The steps

1. **Probe the layers.** Read every layer's state on held-out conversations, and fit a linear top-1 for tokens t+1 to
   t+3. Tap where the next tokens sit. On Kolibri that's 44-48 of 50, not the middle; three taps added 1.5-2 points
   over one. Minutes on the GPU.
2. **Pick the draft vocabulary from the target's own output.** Count the target's top-1 tokens over its recorded
   output per class, and take the slice that covers what you serve. A token outside the slice can never be drafted,
   so coverage caps acceptance. On Kolibri, 32k covered German at 0.896; widening to 64k (every token seen) gave
   +0.075 / +0.014 / +0.037 accepted (code / chat / German) without retraining. A wider slice can be scored before
   any training.
3. **Train on the target's distribution, not on text.** The loss is KL to the target's top-k at every position, with
   the target's states as inputs (prefill any text through the target). On the same 10M tokens it gained about 2x
   what L1 state regression with cross-entropy gained, and trained 17 % faster.
4. **Get volume and variety first.** 45M tokens of broad regenerated text generalised. About 1M rows of the target's
   own output, repeated, were memorised and moved nothing on unseen prompts. Never more than two passes over one
   recording.
5. **Then add the target's own served output.** Serve prompts with the recorder on, and mix the recordings into every
   batch at a fixed share (40 % on Kolibri, code 0.3 / chat 0.4 / German 0.3). On Kolibri, German rose 14 % and chat
   7 % on one chain drafter. When the recordings run out, code acceptance falls back toward the text-only level, so
   end a run with them.
6. **Keep every language and class in every batch.** One 26M-token batch without German cost 18 % of German
   acceptance.
7. **Draft a block, with a chain head at position 1.** A 4-layer block drafter proposing four positions in one pass,
   whose row 0 is an EAGLE-3 chain head, against a 1-layer chain drafter: accepted of 3 went from 1.38 / 1.23 / 0.95 to
   1.91 / 1.29 / 1.11. Start the chain head from a trained chain drafter.
8. **Train position 1 at a low learning rate.** Frozen, it capped the block. At the full rate it forgot its chain
   skill. At 0.1x in its own optimizer group, with loss weight 2, code went from 1.50 to 1.62 in 1,621 steps.
9. **Add a predecessor head.** Each position j >= 2 reads its predecessor token through a low-rank head (rank 256,
   identity at the start). It came with the recordings in one step, so its own share isn't separated.
10. **Serve the body in FP8 and the head slice in NVFP4, and draft every stream in one pass.** On the GB10 a draft
    round costs the bytes it reads, not its launches. The bf16 block drafter was slower than the 1-layer chain and
    lost 17 % at four streams. With an FP8 body, an NVFP4 head (the same drafts, a quarter of the bytes) and one batched
    pass for a round's streams, it is 1.10x at four streams. Draft depth 2.

## Don't bother

- **Loss variants past step 3.** KL, a soft-target CE and AUF tied on Kolibri.
- **A teacher-to-self schedule for the predecessor.** Under greedy acceptance a drafted predecessor survives only if it
  equals the target's argmax, so on every accepted prefix the two are the same token.
- **The data token as predecessor, and a DSpine-style spine.** Both within +-0.002 of the control. We kept the spine
  because it costs little.
- **CUDA graphs and in-place cache rows for the draft pass** alone: no gain until the bytes per round came down.
- **Depth 4 on a MoE target**: slower than depth 2. Each extra verify row brings its own experts.

## Operating notes

- Generation and training can't share a 128 GB machine with a ~77 GB target. Alternate them, and keep each recording
  set's scoring split fixed.
- The recording server needs at least as many parallel slots as clients, or the prompt cache misses and every agent
  turn prefills from scratch (on Kolibri, 135 of 2.8M prompt tokens hit with 7 clients on 6 slots; 95 % with 8).
- Code acceptance scored on repositories the drafter trained on is optimistic. Keep the scoring repositories unseen.
