# NVFP4 V2 + V1-Requantization — coexistence scaffolding, grouped Triton stubs, test harness

## Goal

Land the *structure* for the two new NVFP4 training recipes from
`~/monorepo/cutile/nvfp4_v2_triton/docs/nvfp4_v2_recipe_design_doc.md`, alongside the
shipped V1 recipe. Kernel bodies are deliberately left unwritten — the user writes
the Triton.

## Current state: complete and verified

Everything below the `@triton.jit` bodies is written and exercised. Each of the 7
new kernel files has full validation, output allocation, grid, launch, and
`register_fake`; the body is `tl.static_assert(False, "<kernel> is not implemented")`.

**Key design property, verified bitwise:** a dense linear *is* the degenerate
`num_tensors = 1` case of a grouped kernel. Confirmed on the shipped kernels —
`triton_group_rht_amax` / `triton_group_rht_quantize_row_col` at `num_tensors=1`
produce codes, scales and amaxes bitwise identical to the linear
`triton_rht_amax` / `triton_rht_quantize_row_col`. There is therefore **no linear
kernel set** for V1_REQUANT or V2, now or later.

### Files

| File | What |
| --- | --- |
| `nvfp4_recipe.py` | `NVFP4Recipe` enum (V1 / V1_REQUANT / V2), `NVFP4_CAST_NUMERATOR`=2688, `EDEN_NUMERATOR`=1536, `_amax_to_scale` |
| `nvfp4_rht_cadence.py` | `resample_nvfp4_rht_signs(model, seed, step, microbatch)` — in-place, deterministic, per-FQN |
| `nvfp4_linear_v2.py` | `_NVFP4LinearV2`, `_NVFP4LinearV1Requant` + wrappers, `_degenerate_group_args` |
| `nvfp4_grouped_mm_v2.py` | `nvfp4_v2_grouped_mm`, `nvfp4_v1_requant_grouped_mm` |
| `group_row_cast_quantize_triton.py` | §11.1 **stub** |
| `group_col_cast_requantize_triton.py` | §11.6 + §11.7 **stubs** (share `_load_requant_weight_tile`) |
| `group_col_rht_requantize_triton.py` | §11.4 + §11.5 **stubs** (share `_load_rht_requant_weight_tile`) |
| `group_row_cast_col_rht_amax_triton.py` | §11.8 **stub** (RHT-128, dynamic signs) |
| `group_row_cast_col_rht_quantize_triton.py` | §11.9 **stub** |
| `group_row_rht_col_rht_amax_triton.py` | §11.2 **stub** |
| `group_row_rht_col_rht_quantize_ms_eden_triton.py` | §11.3 **stub** |

Modified: `hadamard_utils.py` (H128 via Kronecker, `get_dynamic_rht_matrix`,
`_nvfp4_global_scales` / `_rescale_fp4` / `_load_scales_swizzle` /
`convert_4xfp4_packed_to_8xfp32` ported from the monorepo, tensor guard on
`get_rht_matrix`), `group_hadamard_utils.py` (`rht_size` param,
`_validate_requant_weight_inputs`), `nvfp4_training.py` (`recipe` field + dispatch),
`nvfp4_single_gpu_example.py` (`--recipe`).

torchtitan: on branch `nvfp4_v2`, cut from `nvfp4_moe_converter` (**not** from the
detached HEAD this session started on). That branch **already had** a working
`NVFP4GroupedExperts` + `NVFP4GroupedExpertsConverter`, better integrated than the
version drafted here — it follows torchtitan's meta-init buffer protocol
(`register_buffer(None, persistent=False)` + `_init_self_buffers`), which the draft
skipped. The draft was discarded and the existing class **extended** instead:
`fc1_recipe` / `fc2_recipe` (both defaulting to `"v1"`), independent FC1/FC2 seeds
and sign vectors, and a `forward` override that delegates to the parent whenever
both recipes are `"v1"` — so the default configuration still reaches NVFP4 through
the `_grouped_mm` seam and the DTensor/spmd preamble is not duplicated.
`__init__.py` and `utils.py` needed no change; that branch already had both.

### Verification

- `pytest test/prototype/moe_training/nvfp4_training/ test/prototype/moe_training/test_nvfp4_grouped_mm.py`
  -> **826 passed, 112 skipped, 0 failed** (13m35s on a GB200). Baseline before this
  change was 740 passed / 37 skipped, so the V1 regression is intact and the
  `_nvfp4_quantize` -> `_nvfp4_global_scales` refactor is bitwise-neutral. The 112
  skips are the numerics tests waiting on kernel bodies.
- `--recipe v1` example still runs end-to-end at DeepSeek-V3 shapes (128 experts, 7168 dim).
- `--recipe v1_requant|v2|moe_split` reach their first unimplemented kernel — wiring proven.
- 30+ new wrapper-layer tests pass today (validation, `register_fake`, cadence, RHT matrix).
- `ruff check --select F,I` and `ruff format` clean.

## What changed since cb564fac

**Removed the stochastic-rounding arm of §11.9.** `enable_stochastic_rounding` /
`rng_state` on `triton_group_row_cast_col_rht_quantize` had no caller: both V2
forwards (`nvfp4_linear_v2.py`, `nvfp4_grouped_mm_v2.py`) passed `None, False`,
because V2's backward gradient goes through MS-EDEN (§11.3) rather than through
FP4 SR. The flag was copied from the shipped V1 op `triton_group_rht_quantize_row_col`,
where it *is* load-bearing -- `nvfp4_grouped_mm.py` calls it with `False` in forward
and `True` in backward -- and that op is untouched, as is `_validate_rng_state`
(still used by §11.3 and the V1 path).

Dropped from the schema, `register_fake`, the no-Triton fallback, the kernel
signature (four seed/offset pointers + the `STOCHASTIC_ROUNDING` constexpr) and the
autotune key, which is now `("N", "FAST_MATH")`. Three tests deleted:
`test_stochastic_rounding_is_reproducible_and_offset_sensitive`,
`test_rtne_and_sr_agree_on_exactly_representable_input`,
`test_stochastic_rounding_requires_an_rng_state`.

Design doc §3 is therefore unimplemented by choice, and the kernel docstring says so.

### Preflight

Contract: remove the dead SR flag from §11.9 only. No kernel body written, no other
op's schema touched, no behavior change for V1 or V1_REQUANT.

### Reference gap, revised

The four requantize oracles (§11.4-§11.7) already exist in `nvfp4_reference.py` and
are asserted on by their test files. **MS-EDEN (§11.3) is the only missing
reference.** Writing it needs, in order:

1. `fp8_max` threaded through `nvfp4_reference_quantize` -> `_block_scale` (it is
   hardcoded 448 there; `global_encode_scale` already takes the parameter but the
   caller drops it). Verify bitwise-neutral at the 448 default before building on it.
2. The Eden correction, per 1x16 block:
   `ratio = sum(scaled^2) / sum(scaled * dequant(codes))`, falling back to `1.0` when
   `dot_cross == 0` or the ratio is non-finite; `sf *= ratio`.
3. The SR of the corrected scale to E4M3 -- do **not** replicate Philox. Return the
   pre-SR corrected fp32 scale and assert the kernel's byte is one of its two E4M3
   neighbours. That pins steps 1-2 bitwise and bounds step 3 to two admissible values.

### CUDA ground truth (~/kitchen), for the record

Only two stubs have a single-kernel counterpart: §11.1 ->
`quantize_transpose_vector_blockwise_fp4` (`return_identity=True`), and §11.3's
MS-EDEN epilogue -> `quantize_transpose_vector_blockwise_fp4_eden`. §11.2/§11.8/§11.9
must be composed from `hadamard_transform_sm90_plus[_amax]` plus a quantize op --
and at dimension 128 kitchen returns only *one* of identity/transposed per call, so
§11.2 needs two. The fused RHT+quantize kernel `quant_nvfp4_optionally_hadamard16`
is RHT-16 only (uint16 sign mask) and is not a reference for any V2 stub. §11.4-§11.7
have no kitchen kernel at all: nothing there consumes packed FP4 as input.

Note the monorepo's V2 casts with **four-over-six**; torchao's does not
(`NVFP4_CAST_NUMERATOR = 448*6`). `test_cuda_equiv_four_over_six.py` is therefore not
a spec for §11.1 or §11.9.

## Next action

Write the `@triton.jit` bodies, starting with **Phase A** (no RHT, unblocks
V1_REQUANT end-to-end): §11.1, then §11.6/§11.7. Each stub's docstring carries a
sketch of the intended body and the specific hazards (int64 widening, NaN
re-injection, bf16 round-through placement).

Then flip `_KERNEL_IMPLEMENTED = False` → `True` at the top of the matching test file:

| Kernel | Test file |
| --- | --- |
| §11.1 | `test_group_row_cast_quantize.py` |
| §11.6, §11.7 | `test_group_col_cast_requantize.py` |
| §11.8, §11.9 | `test_group_row_cast_col_rht.py` (two flags) |
| §11.2, §11.3 | `test_group_row_rht_col_rht.py` (two flags) |
| §11.4, §11.5 | `test_group_col_rht_requantize.py` |
| integration | `test_nvfp4_linear_v2.py`, `test_nvfp4_grouped_mm_v2.py` |

MS-EDEN (§11.3) additionally needs `stochastic_rounding_fp8_e4m3` and
`_quantize_ms_eden` ported from `cutile/nvfp4_v2_triton/kernels/hadamard_utils.py`
(monorepo lines 249 and 638). They are the algorithm rather than plumbing, so they
were left with the kernel body.

## Three corrections to the design doc, applied

1. **`MAX_GROUPS = 64` is CuteDSL-only.** It comes from a depth-6 unrolled binary
   search. Triton's `_get_group_idx_binary` is an unbounded `while` loop, so the new
   Triton wrappers carry **no** such guard — adding one would reject valid 128-expert
   DeepSeek shapes the existing Triton grouped path already handles.

2. **§9's "requant amax generally differs from `amax_w`" is false**, and a test
   written that way fails. When `global_amax` is the tensor's own amax the two are
   **bitwise equal**: the element attaining the max sits in a block whose block amax
   *is* that value, so its block scale saturates at exactly 448, its code is exactly
   6, and it dequantizes back to exactly `amax_w`. The honest discriminating input is
   an **over-bounding** amax (TP all-reduced, or stale). Verified numerically; the
   test in `test_group_col_cast_requantize.py` uses the corrected form and carries
   the explanation.

   Related: §10's "exactly representable → idempotent" needs the *block scale* to be
   E4M3-exact too, not just the values. A tensor drawn from the FP4 value grid gives
   50% of elements differing. `_representable_weights` uses constant magnitude 6 with
   random signs, which makes every block scale exactly 448.

3. **Passing a tensor to `get_rht_matrix` does not raise** — the doc predicts
   `TypeError: unhashable type: 'Tensor'`, but tensors hash by *identity*. It
   silently caches, and because V2 updates its buffers in place the id never changes,
   so it returns the matrix built from the buffer's **first** contents forever: the
   transform stops cancelling and gradients are silently wrong. Added an explicit
   `isinstance(sign_vector, torch.Tensor)` guard with a pointer to
   `get_dynamic_rht_matrix`, plus a regression test.

## Environment trap worth knowing

This checkout has **two torchao installs**: the working copy at
`third_party/torchao`, and an older one in
`/usr/local/lib/python3.12/dist-packages/torchao`. The dist-packages copy shadows
the working copy from any cwd outside the torchao repo, and it predates
`nvfp4_grouped_mm`. In torchtitan this makes the guarded torchao import block fail,
which silently sets `NVFP4Linear = None` and **skips every NVFP4 test** rather than
failing. Run torchtitan with `PYTHONPATH=/home/me/nvfp4/third_party/torchao`, or
install the working copy editable.

Installed this session to make the torchtitan suite runnable: `tyro`, `spmd_types`
(0.2.3 per `pyproject.toml`, not the 0.2.1 in `requirements.txt`), `torchdata`,
`datasets`, `tensorboard`, `wandb`, `tokenizers`, `safetensors`, `einops`, `pillow`.
Before those, 23 of the 33 quantization tests failed on missing dependencies alone.

## Open item — V2 sign-resample cadence in torchtitan

`resample_nvfp4_rht_signs(model, seed=..., step=..., microbatch=...)` exists and is
tested, but **nothing in torchtitan calls it yet**. It needs a trainer hook that
knows both the optimizer step and the microbatch index; torchtitan does not expose
one today. Until it is wired, a V2 run uses the initial sign draw for the whole run —
correct, but forfeiting the variance reduction resampling buys. This is the one piece
of §15/§17 that is scaffolded rather than finished.

## Also not done (deliberate)

- CuteDSL kernels for the new recipes (Triton only).
- Tensor-parallel V2 — `NVFP4Linear.forward` raises for non-V1 recipes with a
  `process_group`, since torchtitan does not use TP NVFP4 linears.
- `benchmarks/prototype/nvfp4_training/bench_group_*.py` for the 7 new kernels; repo
  convention expects one per kernel, deferred to the implementation change.
- Wiring `resample_nvfp4_rht_signs` into a torchtitan training loop (see the open
  item above).

## Surgical Simplicity

- `nvfp4_recipe.py` — the numerator constants exist so every `_amax_to_scale` call
  site names its own; the failure mode (wrong numerator per operand) is silent and
  costs ~40% on backward.
- `_degenerate_group_args` — one `lru_cache`d helper, 2 call sites, load-bearing for
  `torch.compile` stability. The entire cost of the linear-as-one-group choice.
- `_v2_marks.py` — shared skip marks, 7 test-file reuse.
- 4 device helpers ported into `hadamard_utils.py` — without them 4 of the 7 stubs
  cannot be filled at all. `_nvfp4_quantize` was refactored onto `_nvfp4_global_scales`
  so the requantizers decode with the same code the forward encoded with, which the
  design doc calls a correctness requirement; verified bitwise-neutral by the V1 suite.
- Guard on `get_rht_matrix` — prevents a silent wrong-answer path with a concrete
  caller (V2), not a hypothetical.
