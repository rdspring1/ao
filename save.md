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
| `group_row_cast_quantize_triton.py` | §11.1 **implemented** |
| `group_col_cast_requantize_triton.py` | §11.6 + §11.7 **implemented** (share `_load_requant_weight_tile`) |
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

### MS-EDEN reference: written

`reference_ms_eden` now lives in `nvfp4_reference.py`, returning the deterministic
half of MS-EDEN -- RTNE codes, the pre-correction block scale, the corrected fp32
scale, and `ideal_dequant`. `fp8_max` is threaded through
`nvfp4_reference_quantize` -> `_block_scale` (448 default, verified bitwise-neutral
by the 825-test suite). The stochastic draw is **not** reproduced: the monorepo's
`_philox4x32_10` reference was deliberately not ported, and the SR step is instead
bounded by a converging hypothesis test.

Four §11.3 tests changed, all gated behind `_MS_EDEN_IMPLEMENTED`:

* `test_ms_eden_is_unbiased` -- retargeted. It asserted `E[dequant] == dy @ R_n`,
  which is a property MS-EDEN does not have: the codes are RTNE, so the only unbiased
  step is the E4M3 rounding of the corrected block scale, and the expectation
  converges on the Eden-corrected reconstruction instead. Measured at 128x256, the
  old assertion would have *passed* at 0.0693 against its 0.02*max bar of 0.0869 --
  a 20% margin on a false claim, equally likely to have become a spurious failure on
  a different input. Now bounded at `5*SE + 1e-5*norm`, which tightens with `draws`.
  It also passed `is_swizzled=False` where §11.3 returns swizzled scales.
* `test_codes_are_rtne_from_the_pre_correction_scale` -- new. The assertion that
  separates MS-EDEN from ordinary FP4 SR; a kernel that reached for
  `_pack_fp4(..., STOCHASTIC_ROUNDING=True)` passes everything else in the file.
* `test_each_rng_slice_drives_exactly_one_output` -- new. `rng_state` packs both axes
  into one tensor, so a crossed slice correlates the two operands silently.
* `test_codes_per_group_isolation` -- new, and the only multi-group numerics test
  §11.3 has. Scales cannot be compared across group counts (the Philox counter is a
  global element index), but codes can.

Also ported `test_resampled_signs_change_the_output_without_retracing` (§11.2) from
the monorepo: V2 resamples sign buffers in place, and neither failure mode -- graph
retraced per microbatch, or the first draw baked in and cancellation lost -- raises.

### Grouped references: assembled buffers, not per-group pairs

`reference_group_row_cast_col_rht_quantize` (§11.9) and the new
`reference_group_row_rht_col_rht_quantize_ms_eden` (§11.3) return the kernel's
whole-buffer shapes. They have to: the columnwise scale buffer puts the grouped token
axis on the inner, 64-blocked side, so its swizzle tiling restarts at every group
boundary and only the assembled byte sequence means anything. Per-group pairs are why
§11.9's test previously asserted rowwise codes and nothing else -- three of its four
outputs, including that columnwise scale buffer, were unchecked.

§11.3's scale bytes are one stochastic draw from the reference, so they are bounded
rather than matched: `assert_scales_adjacent(..., max_ulps=1)`. SR always lands on one
of the two E4M3 neighbours of the value it rounds and positive E4M3 bytes are
magnitude-monotonic, so one ULP is exact. The bound is loose on value and tight on
*position* -- a scale written to the wrong offset lands nowhere near its neighbour
pair -- which is what makes it a swizzle-layout guard without touching Philox.

`from_blocked_grouped` now lives beside its `to_blocked_grouped` inverse in
`nvfp4_reference.py`; it replaced an identical private copy in
`test_group_rht_quantize_row_col.py` and that file's five call sites now import it.

### Note for whoever pastes §11.8 / §11.9

Their bodies are copies of the shipped V1 kernels. `_group_rht_amax_triton_kernel` is
already generic -- it takes `RHT_SIZE: tl.constexpr` and the wrapper derives
`rht_size = B.shape[0]` -- so §11.8 is verbatim. `group_rht_quantize_row_col_triton.py`
is *not*: lines 126 and 131 hardcode the RHT size. Those are the only two RHT-sized
literals in the file; every other `16` is the NVFP4 1x16 scaling block, which stays 16.
Change one and not the other and you get a well-formed tensor full of garbage, caught
by the columnwise assertions added above and by nothing else.

RHT-128 is intrinsically 8x the transform FLOPs of RHT-16 per tile (128/16). That is
the recipe's cost, not an implementation artifact, and CuteDSL will not recover it.
What may need retuning is register pressure from the 128x128 bf16 fragment.

### Everything else in the monorepo suite is already covered

Cancellation (x3) and the RHT matrix properties are in `test_rht_matrix.py`, which
also has tests the monorepo lacks; forward/backward quality is in
`test_nvfp4_linear_v2.py`. Not applicable: TP V2 (torchao raises by design) and
everything four-over-six. `test_amax_rht_x_t_differs_from_plain_amax` is redundant
against a bitwise reference match.

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

## Phase A is done: V1_REQUANT runs end to end

§11.1, §11.6 and §11.7 are written and green. **853 passed, 88 skipped** across
`nvfp4_training/` + `test_nvfp4_grouped_mm.py`, up from 825/116 -- 28 tests moved from
skipped to passing and the totals reconcile exactly, so no V1 regression. Flags now
`True`: `_KERNEL_IMPLEMENTED` in `test_group_row_cast_quantize.py` and
`test_group_col_cast_requantize.py`, `_V1_REQUANT_KERNELS_IMPLEMENTED` in both
integration files. `_V2_KERNELS_IMPLEMENTED` stays `False`.

V1_REQUANT is ready for the DSV3 671B 300-step convergence run. Decide the acceptance
criterion first: it will **not** match V1 step for step, because the weight path moves
from 16x16 2D scaling to 1x16 rowwise plus lazy requantization -- that is the recipe,
not a bug. Pick a final-loss delta or a curve-divergence band before starting or the
result is not decidable.

### Bugs found reviewing the first implementation

Recorded because the same shapes will recur in Phase B. Two were silent, the rest
were trace-time NameErrors:

* **Missing expert offset on the decode amax.** `tl.load(amax_w_ptr)` with no
  `+ expert`, so every expert decoded `W_qdq` with expert 0's global amax. Fixing the
  NameError alone leaves it compiling and wrong. `test_requant_amax_per_expert_isolation`
  is the guard.
* **NaN probe on the reduced scalar.** The re-injection tested `amax_dw_t != amax_dw_t`
  after `tl.max` had already stripped the NaN. It has to test the tile.
* `tl.atomic_max` given a loaded value instead of a pointer; `_nvfp4_global_scales`
  called without its `FP8_E4M3_MAX` constexpr (it has no default); leftover
  `qa_t_ptr` / `sfa_t_ptr` plumbing from `group_quantize_2d`, which §11.1 has no
  transposed output for; `convert_8xfp32_to_4xfp4_packed` used but not imported in
  either file.

### One test of ours was wrong, not the kernel

`test_rowwise_error_is_no_worse_than_the_2d_scheme` asserted 1x16 error <= 16x16
**per block**. False: about one block in six is worse, because a 1x16 scale is finer
but still rounds to E4M3 and a coarser scale can land on a luckier byte. The claim
that holds -- and the one justifying the extra backward pass -- is the aggregate, near
82% of the 2D error across five seeds. Retargeted to `err_1d.sum() < 0.95 *
err_2d.sum()`.

## Next action

Phase B: §11.8 and §11.9 first (copies -- see the paste note below), then §11.2, then
the real work in §11.4/§11.5 and §11.3. Flip `_AMAX_IMPLEMENTED` /
`_QUANTIZE_IMPLEMENTED` in `test_group_row_cast_col_rht.py`, `_AMAX_IMPLEMENTED` /
`_MS_EDEN_IMPLEMENTED` in `test_group_row_rht_col_rht.py`, `_KERNEL_IMPLEMENTED` in
`test_group_col_rht_requantize.py`, and `_V2_KERNELS_IMPLEMENTED` in both integration
files.

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
- `benchmarks/prototype/nvfp4_training/bench_group_*.py`; repo convention expects one
  per kernel. Deferred to the implementation change, and still owed for §11.1, §11.6
  and §11.7 now that those have landed.
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
