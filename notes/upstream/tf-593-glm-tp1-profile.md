POSTED 2026-10-14 #593 issuecomment-6163729754 (user "ok"); build disclosed as 5da1e007 (four tiles, older base), decode-check gone on 9a099a7c

Method: nsys -t cuda (exllamav3 with --cuda-graph-trace=node), dc.json `code` prompt (2,998 ids), 8 vs 128 greedy
tokens, per-token = (128 run - 8 run) / 120. TF = BHCC glm-exl3-wip (9a099a7c content) + mtp-shard loader patch,
glm-decode-check plain. exllamav3 v1.6.1 same checkpoint. GB10. Unprofiled: TF 21.2 tok/s (47.2 ms), exl3 24.0 (41.7).

| per decode token        | TF ms (launches) | exllamav3 ms (launches) |
| EXL3 projections        | 20.42 (518: tf_exl3_tc_2 259 + rotate_f 259) | 16.76 (133: exl3_gemv_int8_sq) |
| routed experts          | 13.92 (84)       | 16.21 (168)  |
| attention / KDA / DSA   | 4.89 (271)       | 3.62 (223)   |
| hc / norms              | 1.22 (293)       | 1.47 (417)   |
| router / top-k          | 1.18 (95)        | 0.87 (95)    |
| other                   | 0.87 (484)       | 2.06 (233)   |
| kernel sum              | 42.50 (1745)     | 40.99 (1269) |
| wall                    | 47.15            | 41.7 (graphs) |

Reading: TF's experts are 2.3 ms faster; the gap is (1) ~4.7 ms of launch gaps per token (no CUDA graph on the GLM
decode path, 1,745 launches) and (2) projections 3.7 ms slower: 259 calls + a separate rotate each, vs exllamav3's 133
fused calls (its int8-sq GEMV does the Hadamard inside and merges projections). Both close -> ~38.8 ms, ~25.8 tok/s.
MTP already 26.8-30.6 tok/s, ids == plain.
