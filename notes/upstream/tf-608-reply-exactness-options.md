# #608 reply draft: Triton exactness, options A and C

POSTED 2026-10-13 https://github.com/ashhart/TensorFold/pull/608#issuecomment-6140163088 (user: "ok").

---

@BobClawblaw, thank you, that harness is exactly what was missing. I reproduced it on a GB10 (sm_121) against our own
captured set: the same counts, 5 of 26 outputs bit-identical.

Reading the captured kernels' PTX and TTGIR, the differences that remain come in two kinds:

1. **Arithmetic, the same in every kernel:**
   - `exp` is `ex2.approx` of `x * log2e`;
   - `sigmoid` and `/` are `div.full`;
   - `1 / sqrt` is `sqrt.approx.ftz` then `div.full`, and DFlash2's `rsqrt` is `rsqrt.approx.ftz`;
   - normed q and k are rounded to bf16 before RoPE;
   - in `a*b ± c*d`, LLVM fuses the first product and rounds the second.
2. **Reduction trees, one per kernel.** Each sum follows that kernel's blocked layout:
   - `_add_rmsnorm`: 8 warps × 8 values a thread per pass, then 3 shuffle steps across the warps;
   - `_attn_prep`: 2 warps × 4;
   - `_prep_kernel`: 1 warp × 2 per half, then the halves added;
   - `_gdn_pre`: 2 warps × 2;
   - `_gated_norm` and `_gate_mul`: 4 values a thread, 16 lanes.

   Inside a thread the first square is fused with the second (`fma(x0, x0, x1*x1)`).

   The attention kernels add Triton's tile, `ex2` and online-softmax order on top.

Your run shows that kind 1 alone does not pass the gate: one element 1 ulp off in one of 64 layers is enough to
change a near-tie. That leaves two ways forward. @ashhart, which bar do you want for the native glue?

**A. Bit-exact with the captures.** Each shared module gets a Triton-matching arithmetic option (kind 1), plus kernels
that copy each Triton kernel's thread layout (kind 2), checked by `glue-diff` to 0 differing elements.

- For: #608 keeps its current gate unchanged (serial, windows, drafted and multi equal 0.6.6). Any later
  difference is a bug by definition, and no judgement is needed.
- Against:
  - The shared modules carry one reduction order per Triton kernel they replace. `qk_rows`, for example, needs one
    order for `_attn_prep` and another for `_prep_kernel`. Shared code grows per-model branches.
  - The orders come from one Triton version's compiler choices: the layouts, `num_warps`, LLVM's FMA contraction.
    A new Triton or a retuned `num_warps` changes them, and the copies are then exact with nothing.
  - Attention is the hard part: tree attention and block attention would have to copy Triton's flash-attention
    tiles and softmax rescaling bit for bit. That takes much longer than the eight glue kernels (about a day).
  - The approximate `ex2`, `div.full` and `sqrt.approx` stay forever, in code that otherwise holds every element to
    a float64 bound.

**C. The native kernels become the 27B's reference.** Exactness is defined against the native kernels themselves,
and their accuracy against float64:

- each kernel within its float64 tolerance at the 27B's shapes, with a mutation it catches (as now);
- the engine's own invariants: drafted == serial, concurrent == alone, resumed == fresh (they hold today: 12/12
  `bench_concurrent --alone --serial` cells and drafted == serial on all four prompts);
- quality against the captures, teacher-forced on real text: NLL 0.8058 vs Triton's 0.8029, argmax agreement
  99.12 %, mean KL 0.0012;
- `glue-diff` stays as the evidence that every difference is a bounded ulp-level one, not a bug.

- For:
  - One arithmetic for every family. Nemotron, the 27B and later families share the same kernels and the same
    test bar.
  - No dependence on a Triton version.
  - The kernels are already written and tested.
  - Tree and block attention stay as they are, and they are faster: attention 46.1 vs 49.0 ms on the drafted long
    prompt.
- Against:
  - Two of four replies differ from 0.6.6 at near-ties (an exact top-2 tie, and one ulp). So 0.6.6 and native
    are no longer interchangeable byte for byte, and "exact against 0.6.6" becomes "exact against itself, close to
    0.6.6".
  - A real regression hides more easily inside "close" than inside "equal". The NLL/KL/argmax numbers have to be
    rerun for each change that touches the glue.
  - It needs your sign-off, since it changes the acceptance rule #608 was written against.

My preference is C. If you'd rather keep the 0.6.6 bar, I'll do A for the glue kernels first and report how far
attention gets.
