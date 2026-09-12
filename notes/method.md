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

## Name the differing cell before launching an A/B

Before starting any two-arm run, write in the runner's own header: **which measured cell must differ
between the arms, and by what mechanism.** If no cell in the benchmark's case list can differ, the
run is vacuous and must not be started.

Checking that the *knob* moved is not a mechanism check. Three vacuous runs in one week, all mine,
all passing a knob-level check:

| run | knob verified | why it could not reach the measurement |
| --- | --- | --- |
| `cgsize2` | capture size env set | no cudagraphs were captured in *either* arm |
| `cgnone2` c>=4 | `maxcap` logged 8 vs 0 | MTP-3 makes the batch 4*c > 8, so both arms ran eager |
| `thr` | `RADIX_THRESHOLD` differed in each build | every benched width sat on the same side of all three thresholds |

The check that would have caught all three is the same one: take the benchmark's actual case list,
and for each arm compute which code path each case takes. If the two columns are equal, stop.

## The cell must be the ONLY thing that moved

Naming the differing cell is not enough — the rule as written asks whether the knob can *reach* the
cell, and that is satisfiable by a cell containing three other changes. On 2026-09-12 a five-rung
bisect ran against "the serving venv" while that cell also held **our own tile-union prefill kernel**
(on in one venv, absent in the other), **our #55715 GDN patch** (the reverse), and **torch.compile
on versus off** (upstream #55272 removed the decorators). Two of those were ours and invisible from
the launcher.

So before an A/B, diff **our own patch markers** across the arms, not just the version strings:

```bash
for m in "QSADET (jschmied" "DETFIN (jschmied" "Clear any semaphore still raised" \
         "GDN55715-PATCH" "tile_union" "jschmied"; do
  printf "%-34s A:%s B:%s\n" "$m" "$(grep -rl "$m" "$A" --include=*.py | wc -l)" \
                              "$(grep -rl "$m" "$B" --include=*.py | wc -l)"
done
```

Any row where the counts differ is part of the cell whether you meant it to be or not. A marker that
is *present* is not necessarily *active* — check the run log too; the PLE mmap overlay was in one venv
and inert because its gate needs `--enforce-eager`.

## State no cause until the alternative is measured

A finding may say "A differs from B". It may not say "A differs *because* of C" until C has been
varied and measured. Reading the dispatch code is not measuring; neither is a plausible mechanism.

Of ~13 errors on 2026-09-12, **ten** were this: a published FlashInfer pin written from a PR
description instead of the wheel; a 13-minute startup blamed on an autotune cache that timing showed
was just the cost of loading 123 GB; "no PTX, so it is an arch gap" when the errno was
`cudaErrorNotSupported` (801) and a missing cubin is 209; "make the clamp unconditional" when the
value being clamped was an allocation size the kernel indexes. Each counterfactual took minutes, and
the data was already on disk in every case.

Practical form: before writing a cause, name the run that would have come out differently had the
cause been false. If there is none, write it as a hypothesis and say so.

## And verify the control arm actually misbehaves

Naming the differing cell is necessary, not sufficient. Five void runs on 2026-09-08 split into two
kinds:

- the knob could not reach the measured cells (`cgsize2`, `cgnone2` c>=4, `thr`);
- the knob was fine but **the control never misbehaved**, so the treated arm's cleanliness meant
  nothing (`vpp4`, `vpp5`).

For any "X fixes it" claim, the run must first show the unfixed arm exhibiting the defect. If it does
not, report that and stop — do not iterate on the stimulus until something moves, which is how a
false positive gets manufactured.

Also: **check the units you claim.** `vpp5`'s prompts were sized by `chars/4` and landed at 1,874
tokens against a stated ~2,600, below the very threshold the run was built to exceed. English prose
here is 5.52 chars/token. Ask the tokenizer, not the estimate.

## Reading a cross-arm quality comparison

`analyze.py` (ZC502's, `/opt/llm/runners/vpp/analyze.py`) takes `reference [candidate]` as **files**
and `--out` as a **directory**. With one file it reports only self-consistency across that arm's
repeats (`reference.summary.positions_with_top1_disagreement`). With **two** files it additionally
writes a `cross_arm` block, which is the one that answers "do these two configurations produce
different output" rather than "is this configuration stable":

- `positions_with_cross_arm_modal_top1_mismatch` — positions where the two arms' *modal* top-1 token
  differs. This survives per-arm jitter, so it is the quality number.
- `max_abs_forced_logprob_mean_delta` — how far apart the distributions get, and where.

`/opt/llm/runners/xarm.py <report-dir…>` prints both blocks for one or more report directories; use
it instead of hand-parsing `report.json`. Every A/B in this repo that changes numerics — a cache
dtype, an approximation like index sharing, a scheduler mode — needs the two-file form, because a
speed win with a moved modal answer is a trade, not a lever.

## A mechanism check must test a string, not a pipeline's exit status

`say "MECH: $(grep -o … "$LOG" | head -1 || echo NOT-FOUND)"` is broken: in a pipeline the exit status
belongs to the **last** command, and `head` always succeeds, so the `||` fallback never fires and the
marker comes back empty. An empty marker then falls through whatever `case` the gate uses and can void
a perfectly good run — it did, on `ishare`, 2026-09-09 (finding 157), which stopped after two of six
arms with the flag correctly engaged. Capture first, then test the string:

    M=$(grep -oE "…" "$LOG" | head -1); [ -n "$M" ] || M="NOT-FOUND"

Same rule for every gate: assert on a value you can print, and print it into the results file so the
harvest can see what the gate saw.

## Record the kernel roster on every run

Read kernel selection **from the run log, never from dispatch source**. Source-reading produced three
wrong claims in one morning (det-180 twice, and the doubt cast on the published Periodic Table), each
refuted in under a minute by a log that had been on disk the whole time.

`/opt/llm/runners/kernelroster.py <server-log> [...]` prints, per log:

- **linear** — `Using <kernel> for NVFP4 GEMM`. *Absence is a result*: it means no quantized dense
  Linear was built, which is what Flash-Next looks like.
- **moe** — the selected backend **and the candidates it rejected**, which vLLM prints for free.
- **attention / vit** — the selected backend and its alternatives.
- **cudagraph** — `GiB for CUDAGraph memory` and the count of `Capturing CUDA graphs` lines, flagged
  `<-- NOT CAPTURING` when both are zero.

Run it on every arm of every A/B and paste the block into the results file. Two arms that differ in a
kernel nobody intended to change are not a comparison, and the roster is the cheapest way to see it.

### Three traps it exists to catch

1. **No `input_scale` tensor does not mean weight-only.** *Dynamic* activation quantization computes
   the scale per token and stores nothing. Counting `input_scale` siblings to infer W4A16 vs W4A4 is
   wrong; read `config_groups[...].input_activations` (`num_bits`, `dynamic`) instead.
2. **The scheme class decides the dispatch.** `compressed-tensors` and `modelopt` reach different
   selectors for the same bit widths. The log line `quantization=<method>` says which one ran.
3. **Match the source to the build.** Logs from vLLM 0.27.1 cannot be explained by reading the
   0.28.1 venv; the registry order and the forcing branches both changed between them.

## Bracket every A/B arm with a bandwidth probe

`/opt/llm/runners/bwprobe.py` **before the arm starts and after it stops** — never while a server holds
the GPU. Tried concurrently 2026-09-11 and it died with a CUDA error at util 0.75; the idle-box
baseline is not comparable to a contended measurement anyway.

GB10's decode-shaped GEMV bandwidth is reported to flip ~70 ↔ ~225 GB/s **without an SM-clock
change**, and `clocks.mem` is `[N/A]` on this hardware — so no `nvidia-smi` field detects it (det-201).
A clock read, which is what the watchdog does, cannot substitute.

Our baseline is **212.8–215.0 GB/s peak at M=1**, 1 % spread. An arm near 70, or one whose opening and
closing figures disagree, is not comparable to one that is not — and a 3× swing dwarfs every effect
this project measures.
