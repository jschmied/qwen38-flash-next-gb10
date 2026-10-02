POSTED on TF #258 (2026-10-02, user: "short reply"): https://github.com/ashhart/TensorFold/issues/258#issuecomment-5959052841

Not intended, it's being fixed. #212 gives EXL3 prompt windows their own expert kernel (each weight tile decoded once
per up to 64 pairs). On one GB10 with the 3.05 bpw pack it took an 8k prefill from 11.4 to 5.3 s and 32k from 46.2 to
21.3 s, same tokens. A follow-up that keeps the next chunk's trellis words in flight (agreed in #212, after it lands)
takes 3.05 another ~14 %: 8k 4.54 s, about 1,800 tok/s. For long cold prompts, #252 reads the n-gram pages ahead. We
haven't measured at 235k, where attention and table paging weigh more.

Written with AI assistance (Claude Code).
