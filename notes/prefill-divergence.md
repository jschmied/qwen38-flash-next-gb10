# The forward pass is not deterministic, and speculation is not the cause

> **CORRECTED 2026-09-01, late.** The title claim is WRONG and is kept only so the correction is
> legible. With `--no-enable-prefix-caching`, three identical single-token requests return
> **bit-identical** logits (`lp=-0.2308074981`, `sig=e22e0de36cac`, 3x). **A single forward pass is
> deterministic.** What diverges are the STATEFUL paths. Read the table below before anything else.
>
> | condition | result |
> | --- | --- |
> | one prefill forward, prefix cache OFF | **deterministic** (1 distinct of 3) |
> | one prefill forward, prefix cache ON | diverges (3 of 3) |
> | 1040-token generation, prefix cache OFF | **diverges** (3 of 3) |
>
> So there are at least **two** things to explain, not one:
> 1. **Prefill divergence requires prefix-cache reuse.** Turning it off removes it entirely.
> 2. **Generation diverges without any prefix cache**, so accumulated decode state is a second,
>    independent path. Speculation is excluded for both (`P_nospec`, `G_eager_nospec`).
>
> Everything below was measured with prefix caching ON (the default) and is still valid as
> measured -- but "the base forward pass diverges" is not a correct reading of it. The 2x2
> factorial and the layer bisection describe the system *with the cache active*.
>
> Not established: whether the two are one defect. A shared cause is plausible (both involve
> reusing state written by an earlier pass) but unproven.

Measured 2026-09-01 on `qwen38-flash-next-fp8head`, vLLM `0.1.dev20073+g8e685d198`, GB10,
batch 4096, `max_model_len` 32768, concurrency 1.

## The result

A **logit probe** — `max_tokens=1`, `top_logprobs=20`, three byte-identical requests sent
sequentially to one server — returns three different logit vectors.

| arm | speculation | top token | logprob of the top token |
| --- | --- | --- | --- |
| `P_ctl` | MTP n=5 | `'#'` | −0.1618698537 / −0.2338832766 / −0.3886644840 |
| `P_nospec` | **off** | `'#'` | −0.3509956896 / −0.2271908522 / −0.2907660007 |

3 distinct of 3 in both arms. The argmax is stable; its **probability moves from 0.85 to 0.68**.
That is four orders of magnitude larger than bf16 rounding on a ~60-token prompt, so this is not
last-bit accumulation noise.

## What it rules out

- **Sampling.** The probe reads logprobs, never samples.
- **The decode loop, spec rollback, KV reuse, feedback.** One forward pass, one token.
- **Speculation entirely.** `P_nospec` has no drafter, no MTP head, no rejection sampler, and
  diverges identically. This is the base forward pass.
- **Carried state between requests.** Divergence appears across three requests inside **one**
  server start, so it proves a **per-request** component. It does *not* by itself eliminate all
  startup state — a per-start component could exist on top of it.
- **The QSA top-k.** `P_ctl` ran with `VLLM_QSA_TORCH_TOPK=1`, substituting `torch.topk` for the
  custom `persistent_topk`, and still diverged 3/3. `persistent_topk` is a real determinism bug
  (vllm#54521) but is **not sufficient to explain this divergence**; at least one further source
  remains. Note the prompt is short enough that QSA barely sparsifies at all, which independently
  points away from the sparse-attention path.

## BISECTED: divergence enters inside layer 1 (2026-09-01)

Forward hooks hashed every decoder layer's output over repeated identical prefills
(`--enforce-eager`, speculation off, `max_tokens=1`). **Passes were grouped by their layer-0
hash** — identical layer-0 output means identical input to the stack, so like is compared with
like. Two independent groups of three passes each, same answer both times:

| module | passes 7, 9, 11 | passes 8, 10, 12 |
| --- | --- | --- |
| `layers.0` | identical x3 | identical x3 |
| `layers.1.ple.ple_embedding` | **identical x3** | **identical x3** |
| `layers.1` | **DIFFERS** | **DIFFERS** |
| `layers.2` .. `layers.47` | differs (downstream) | differs (downstream) |

**Established:**

- **Layer 0 is bit-identical.** Embeddings, the first GEMM, layer 0's GDN attention and its MLP
  are deterministic on identical input. This substantially weakens the CUDA / driver / FlashInfer
  hypothesis: the same kernel families run one layer earlier without a wobble.
- **The PLE gather is exonerated.** `ple_embedding` returns bit-identical output three times in
  both groups. This was the leading hypothesis and it is wrong.
- **Divergence enters inside layer 1**, downstream of the PLE table lookup, within that layer.

**Why layer 1 is special.** `layer_types` makes layers 0, 1 and 2 all `linear_attention`, so the
attention kind is not the difference. Layer 1 is the **only** layer carrying the PLE
(`if self.ple is not None`, `nvidia/model.py:285`; `hidden_states = hidden_states + self.ple(...)`,
:296). So layer 1 == layer 0's structure + the PLE injection, and layer 0 is clean.

**Two candidates remain, and they are separable:**

1. **The PLE path after the lookup** — `qwen3_8_flash_next_ple_short_conv` is a distinct op on the
   PLE path that the hook never observed; only `ple_embedding` was covered.
2. **MoE routing ties** — layer 0's MoE is deterministic *on its own data*, which does not prove
   MoE determinism in general, since routing ties are data-dependent. Compare
   `[[temp0-not-reproducible-under-load]]`, which already suspected MoE routing.

Sub-bisection queued (`bisect2.sh`, hooks every submodule of layer 1) to separate them.

## SUB-BISECTION (bisect2) — and a flaw in the instrument

Hooking every submodule of layer 1, real group [8,10,12] (identical `layers.0` output):

| identical across 3 passes | differs across 3 passes |
| --- | --- |
| `ple.ple_embedding` | `ple.norm_query`, `ple.norm_conv` |
| `ple.key_proj` | `linear_attn.in_proj_qkvz`, `linear_attn.in_proj_ba` |
| `ple.value_proj` | `attn_hyper_connection.input_mix_weight_*` |
| | `linear_attn.chunk_gated_delta_rule`, mlp/MoE, everything downstream |

**Everything derived from the PLE table is bit-identical; everything derived from the hidden state
differs.** That is a clean split and it holds across all three passes.

### The flaw it exposed

`in_proj_qkvz` is a plain GEMM on layer 1's input. **A projection cannot differ if its input is
identical.** So layer 1's input is NOT identical, even though `layers.0`'s output hash is.

The cause is architectural: this model uses **hyperconnections** (`attn_hyper_connection`,
`mlp_hyper_connection`, `GatedResidual`), which carry **multiple residual streams** between layers.
Hashing a layer's returned tensor captures one stream. Two passes can agree on that tensor and
still disagree on the rest of the carried state, which the next layer mixes in.

**Therefore this claim, made earlier and committed, is too strong:**

> ~~"Layer 0 is bit-identical. Embeddings, the first GEMM, layer 0's GDN attention and its MLP are
> all deterministic on identical input."~~

The correct statement is **"layer 0's OUTPUT TENSOR is bit-identical."** The divergence may enter
earlier, in a stream the hook never observed. What the bisection establishes is *the first hooked
module that differs*, which is not the same as where divergence starts. The weakening also partly
restores the CUDA/driver hypothesis, which the earlier phrasing had pushed down.

Not leaned on: the profiling passes [1,2] (zero inputs) show `chunk_gated_delta_rule` differing on
its own, but with zero inputs most modules emit degenerate values and hashes collide across
unrelated modules (`80a3721188` recurs everywhere), so that is suggestive at best.

### The fix

Hash the **hyperconnection streams** themselves — the residual tensors carried between layers —
not each layer's return value. Only that gives a fingerprint strong enough to pin the input, and
without it every layer-level bisection on this architecture will keep answering "layer 1".

## What it does not rule out

The PLE host gather, the embedding lookup, the dense GEMMs, the MoE router, or the CUDA/FlashInfer
layer beneath all of them. Separating these is what `layerhash_patch.py` is for — hash every
decoder layer's output over three identical prefills and find the first layer that differs.

## The remedy is unavailable here

`VLLM_BATCH_INVARIANT=1` refuses to start: no mamba/linear-attention backend implements
`supports_batch_invariance()`, and 36 of 48 layers are linear attention. See
`batch-invariance-unavailable.md`. It would likely have been a null result anyway — batch
invariance addresses variation from batch *composition*, and at concurrency 1 with sequential
identical requests the composition is already constant.

## Why it matters beyond reproducibility

Two observables, one suspected cause:

| | text output | throughput |
| --- | --- | --- |
| no speculation | **diverges** (3 distinct of 3) | **stable** — 1.10× over 12 arms |
| MTP | diverges (3 distinct of 3) | **unstable** — 1.83× over 5 configs |

Consistent reading: divergence lives in the base forward pass. Under no-spec it changes *which*
token is emitted, but every token costs the same, so timing is flat. Under MTP the same
perturbation moves **draft acceptance**, and `ms/tok` is governed by mean accepted length, so it
surfaces as a throughput spread. Acceptance is a matching process — noise can only break matches,
never create them — which predicts the observed **floor** (~31.5 ms/tok, reached by n=3,4,5,6) is
the clean value and everything above it is degradation.

**This is a hypothesis, not a result.** The direct test is queued: five starts of one MTP config
recording `ms/tok` and `mean_accept_len` together (`accepcorr.sh`). A strong negative correlation
confirms acceptance is the channel; a weak one moves the cost elsewhere.

If the mechanism holds and the defect is fixable, the expected gain is **reliability first,
speed second**: median MTP n=5 would move from 58.4 toward its ~32 floor. It would still not beat
`ngram` (floor 28.5) on agent work, and k=2 — which never reaches the floor in any start — looks
structural rather than noise-driven and would probably survive a fix.

## Related

`batch-invariance-unavailable.md`, `failure-modes.md` (the MTP restart-instability section),
`mtp-depth-anomaly.md`, `evidence-standard.md`. Upstream: vllm#54521, vllm#54552.
