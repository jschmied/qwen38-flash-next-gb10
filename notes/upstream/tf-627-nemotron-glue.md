#627 Nemotron CUDA glue on GB10 - POSTED measurement 2026-10-14 (issuecomment-6166128439, user "do 1..3");
ashhart OK'd us taking router/topk/add_moe_norm byte-equal. PR #833 (branch nemotron-route-glue 4f09b51f on main 1e76a70e).

Results (tensorfold run, MLX4 Nemotron, 3 interleaved rounds, data/n627/ab3.log): plain code -2.6 %, prose -1.8 %,
drafted -0.6/-1.6 % vs main; vs Triton set +0.8/+1.6 % plain. Per launch: router 16.4->9.0 (Triton 7.2), topk 7.0->4.3
(3.1), add_moe_norm 8.9->5.5 (3.7). Main-stream wait 1489->730 us/step (Triton 561). Shas equal; teacher 512 steps
byte-identical (local LOCALDIGEST print, not committed); prefill 2k/8k level.
Lessons: ncu Duration useless for tiny kernels (cold caches + overhead) - use nsys in the model; the side stream
(router->topk->plan->routed experts) is the critical path, the main stream (shared expert) waits; glue test is
blind to add_moe_norm's ss order (mutation passes) - teacher-forced logits catch it (508/512).
Left open (said so on #627): one-row attn_chunk (18.7 vs 11.6), attn_merge, group_rmsnorm. Router chain floor:
448 dependent fma per (row, expert, slice) - bits forbid splitting; loads alone ~7.9 us under shared-expert contention.
