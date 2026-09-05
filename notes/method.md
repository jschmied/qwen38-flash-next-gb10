# Method, earned the hard way

<!-- moved out of README.md on 2026-09-05 so the README stays an entry point; content unchanged -->


- **The noise floor is 6.9% — for *decode*** (six identical runs: 34.7–37.1 tok/s). **Nothing under
  ~10% is callable from a single run.** ⚠️ **Prefill is far noisier: ±20%.** Three runs of one
  configuration at 8k input spanned 1,633–2,367 tok/s, a 45% range. A prefill claim needs n≥3 and a
  wider bar than a decode claim; treating the 6.9% figure as general is what produced a withdrawn
  batch-size finding on 2026-08-31. This cost us a published claim — "k=2 is the MTP optimum" compared
  a single k=3 run against the *top* of k=2's own spread. Withdrawn.
- **Verify `lfs.sha256`, not file size.** Two size-correct, byte-corrupt shards produced *fluent
  garbage* invariant to every configuration change, and cost a full day plus two retracted
  upstream issues. `aria2` preallocates, so a file reaches full size the moment it starts.
- **Verify the lever is real at the shape level before building anything.**
  [`tools/shapebench.py`](tools/shapebench.py) takes two minutes and would have pre-empted a
  checkpoint build, four failed server starts and three six-run A/B arms.
- **Prove a kernel actually ran.** Log first-sight dispatch keys inside the op — a call-count
  threshold never fires under cudagraph replay. We peeled back *four* layers of "installed but not
  running" before one measurement meant anything.
- **An all-empty cell is not a comparison.** Twice in one session a determinism test reported
  outputs as *identical* when every response was the empty string — the model was still inside
  `<think>` and the token budget ran out, so five empty strings hashed the same and the check
  passed vacuously. A comparison must assert that it compared something: print the character
  counts, and refuse the verdict when the cell is empty. The second occurrence is the instructive
  one — the guard had already been written into the previous script, and **a guard does not travel
  to the next script you write**. It belongs in the harness, not in one copy of it.
- **Clear `VLLM_CACHE_ROOT` + `TORCHINDUCTOR_CACHE_DIR`** when benchmarking a source-level patch;
  a stale compiled graph silently replays your unpatched code. (Config flags *are* hashed
  correctly — we checked before filing.)
