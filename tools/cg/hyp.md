# Item 3: CUDA-graph capture widths at concurrency, written 2026-09-26 ~15:10, before the run
Capture sizes count TOKENS. With MTP n=3 a decode row is 4 tokens, so today's [1,2,4,8] covers c<=2 only; c>=3
runs the target model eager (piecewise pieces without graphs). Arms (clone venv, prod-like: RecoverSSM on, NVFP4
head, 32k vocab, MTP n=3, align + prefix caching, KV 4 GiB): cg8 [1,2,4,8] vs cg64 [1,2,4,8,12,16,24,32,48,64];
probe decprobe (c=1/c=4/agent) + concprobe (c=4/8/16), 2 starts.
- H-speed: c=1 unchanged (+-1 %); c=4 aggregate +5..+15 % (launch overhead of an eager forward now removed);
  c=8/c=16 +5..+20 %. Agent loop (c=1) unchanged.
- H-cost: graph memory +0.3..+1.5 GiB (log "Graph capturing finished ... took X GiB"); KV tokens at the fixed
  4 GiB unchanged.
- H-correct: hashes identical within each concurrency level between arms is NOT expected (batch shapes change the
  padding/graph path, and greedy is not batch-invariant); c=1 hashes identical.
- Outside: c>=4 slower with graphs -> padding to the next capture size costs more than launch overhead saves.
