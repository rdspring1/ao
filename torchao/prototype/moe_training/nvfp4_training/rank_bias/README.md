# NVFP4 QDQ in CuTe DSL + a standalone `analyze_rank_bias`

A dependency-light port of kitchen's
`experimental/tensor_dump_analysis/analyze_rank_bias.py`. The leaf
quantize/dequantize path is reimplemented twice:

| module | what it is |
| --- | --- |
| `nvfp4_reference.py` | PyTorch oracle, a transcription of kitchen's `nvfp_utils.to_nvfp_verbose` / `from_nvfp_verbose` / `cast_utils` |
| `eden_reference.py` | PyTorch oracle for recipe 100483's MS-EDEN leaf, a transcription of `quantize_transpose_vector_blockwise_fp4_eden.cu` |
| `rht.py` | the Random Hadamard Transform, dim 16 (9004) and dim 128 (100483), a transcription of `perform_random_hadamard_transform_ref` |
| `nvfp4_cutedsl.py` | CuTe DSL kernels (`nvidia-cutlass-dsl`), **bitwise identical** to the oracle |
| `eden_cutedsl.py` | the same for the MS-EDEN leaf |
| `analyze_rank_bias.py` | the sweep, plots and CSV, driving either backend |
| `analyze_sparsity.py` | exact-zero and FP4 flush-to-zero sparsity per tensor / module |
| `test_nvfp4_qdq.py` | the equivalence tests |
| `test_sparsity.py` | tests for the sparsity script |
| `test_kitchen_equivalence.py` | optional cross-check against a *built* kitchen; skipped otherwise |
| `test_psx_clippy_equivalence.py` | optional cross-check against the psx `clippy` CUDA kernels; needs only `psx_formats` |

Nothing imports kitchen, `psx_formats`, TransformerEngine or the rest of the
monorepo. `pip install -r requirements.txt` on any CUDA box (Hopper or newer)
and run.

```bash
RB=torchao.prototype.moe_training.nvfp4_training.rank_bias
pytest -q test/prototype/moe_training/nvfp4_training/rank_bias/

# real dump
python3 -m $RB --tensor-path G_rank0_layer0_fc1_step0.pt \
    --recipes 9004 --variant G.T --trials 2 4 8 16 32 64 128 --save-plot-dir ./plots

# both recipes as two curves per bucket
python3 -m $RB --random 4096x4096 --recipes 6302 9004 --variant G.T \
    --trials 2 4 8 16 --save-plot-dir ./plots

# V1 vs V2 head to head; each recipe uses its own sign lifetime by default
python3 -m $RB --random 4096x4096 --recipes 9004 100483 --variant G.T \
    --trials 2 4 8 16 32 64 --save-plot-dir ./plots
```

## What is modelled

| `--recipes` | G quantizer | randomness on G | W tiles | RHT | signs | variants |
| --- | --- | --- | --- | --- | --- | --- |
| `6302` | NVFP4 | SR on the FP4 data | 1x16 | none | — | all six |
| `9004` (alias `6304`) | NVFP4 | SR on the FP4 data | 16x16 2D | wgrad, dim 16 | static | X, X.T, G, G.T |
| `100483` | MS-EDEN | SR on the E4M3 **block scale**; data is RNE | 1x16 | wgrad **and dgrad**, dim 128 | dynamic | G, G.T |
| `nvfp4` | NVFP4 | none (RNE) | 1x16 | none | — | all six |

`--variant W` / `W.T` under 9004 is rejected rather than silently swept with the
wrong tile shape: the 16x16 `PER_2D_BLOCK` path is not implemented. It is only
W that needs it, so the G analysis is unaffected. `100483` likewise accepts only
`G` / `G.T`; its X and W lanes keep the psx NVFP4 quantizer and would need the
1x16 cast under RHT-128 (`X.T`) and the lazy `col_rht_requantize` path (`W.T`),
neither of which is implemented here.

Kitchen recipe **6302** = `QuantizeRecipe.NVFP4_EMULATION` with `use_sr=True` on
`g_params`, i.e. for every tensor:

* E2M1 data, `quant_tile_shape=(1, 16)`, `ScalingType.PER_1D_BLOCK`
* per-tensor two-level scaling: `global_scale = 6*448 / amax(x)`
* E4M3 block scales, RNE (`ClippyScaleRoundingMode.E4M3_RNE`)
* RNE FP4 rounding on X / W, stochastic rounding on G and G.T
* the emulation op's zero padding — `(32, 16)` on the identity path, `(128, 64)`
  on the transpose path — applied and then trimmed off
* `dequantize()` is kitchen's no-op cast, so the QDQ result is **bf16** whatever
  the input dtype

The transpose path is the identity path applied to `x.T` (16 elements down each
column), which is what the psx `nvfp4_clippy_transpose` kernel does; the CuTe
kernels get there by taking a transposed view of the same memory.

`--recipes nvfp4` is the same recipe with RNE everywhere. It is deterministic,
so the pre-flight stochasticity check rejects it unless you pass
`--skip-stochastic-check` — a flat MSE-vs-trials curve is the expected result.

### Recipe 9004 (kitchen's `config.py` says 6304 is "the exact same as 9004")

9004 is 6302 plus 16x16 W tiles and the **Random Hadamard Transform on the wgrad
lane**. The rotation touches only the *transpose* lane, and only X and G — W is
never rotated, and no identity lane is. So `--variant G` is bit-identical under
6302 and 9004; only `G.T` (and `X.T`) changes:

1. zero-pad rows up to a multiple of 16,
2. rotate: `(x.T.contiguous().view(-1, 16) @ diag(s) @ H16) * 1/sqrt(16)`,
   transposed back — bf16 in, fp32 accumulate, bf16 out, mixing groups of 16
   rows, the same 16 elements the transpose-path QDQ blocks over,
3. the usual NVFP4 QDQ of the rotated tensor,
4. rotate back with `H16 @ diag(s)`, then crop the row padding — kitchen inverts
   first and crops second, and so does this port.

The sign vector `s` is kitchen's checked-in wgrad vector
(`kitchen/ops/data/hadamard_random_sign_vec.json`), because 9004 leaves
`enable_online_randomization` off. kitchen's `--vary-rht-sign` /
`--no-vary-rht-sign` are therefore a **no-op for 9004**. Here the flags do
something: the default is the fixed vector (what 9004 uses, and the only setting
that is bitwise-reproducible against kitchen), and `--vary-rht-sign` re-draws it
per trial from the trial seed, for dumps whose sign vector was randomized online.

**Reading a rotated curve.** Rank labels are computed on the raw tensor, so under
`G.T` a bucket's MSE is the rotated-back mixture of that row-tile's 16
quantization errors, not the error of that element's own block. Kitchen's script
labels the same way, so the curves are comparable — it is the interpretation, not
the number, that changes.

### Recipe 100483 (V2): MS-EDEN, RHT-128, dynamic signs

100483 is not 9004 with a bigger Hadamard. Four things change together
(`kitchen/config.py:10658`):

| | 9004 (V1) | 100483 (V2) |
| --- | --- | --- |
| G data rounding | FP4 **stochastic** (psx `apply_rs`, in-kernel cuRAND) | FP4 **RNE** |
| G block scale | E4M3 RNE | MS-EDEN corrected, then **E4M3 stochastic** |
| decode numerator | `6*448 = 2688` | `6*256 = 1536` |
| RHT | dim 16, **wgrad lane only** | dim 128, **wgrad *and* dgrad** |
| sign vector | static | re-drawn per iteration |

The MS-EDEN leaf, per 16-element block: quantize with the uncorrected E4M3 block
scale, RNE; accumulate `sum_sq = Σ x_s²` and `sum_cross = Σ x_s · code` over the
block with **sequential fp32 FMAs** in element order; take
`correction = sum_sq / sum_cross` (falling back to 1.0 when `sum_cross` is zero
or the ratio is not finite); multiply the scale by it; stochastically round that
to E4M3. The block-scale ceiling is 256 rather than 448 precisely so the
correction, which can only raise the scale, has headroom before it saturates.

**Both G lanes rotate**, and they rotate along different axes with *different*
sign vectors. G's identity lane is the DGRAD GEMM and its transpose lane is the
WGRAD GEMM (`get_gemm_types_for_tensor`), and 100483 enables the transform on
both, so `--variant G` is RHT-128 rotated too. Under 9004 no identity lane was
ever rotated, which is why the recipe table now keys on the GEMM a lane feeds
rather than on "is this the transpose lane".

**Sign lifetime is a recipe property, not a CLI preference.** 9004 leaves
`enable_online_randomization` off and 100483 turns it on, so `--vary-rht-sign`
now defaults to *per recipe* rather than to a single global choice. This is
load-bearing rather than cosmetic: MS-EDEN's FP4 codes are deterministic, so with
the sign frozen the only random quantity left is the block scale, and a trial
mean converges on the *corrected reconstruction* rather than on the input. On a
2048x2048 Gaussian, `G.T`:

| sign | 9004 slope | 100483 slope | 100483 MSE @ T=64 |
| --- | ---: | ---: | ---: |
| forced fixed | -0.90 | **-0.04** | 1.21e-02 |
| forced varying | -0.99 | -1.00 | 2.47e-04 |

Frozen signs make 100483 look like it has no averaging benefit at all. Passing
`--vary-rht-sign` or `--no-vary-rht-sign` still forces one choice on every recipe
in the run, which is what you want to isolate the sign lifetime and *not* what
you want for a faithful 9004-vs-100483 comparison.

## Sparsity: what NVFP4 actually discards

`analyze_sparsity.py` is deliberately a separate script. `analyze_rank_bias.py`
is a *trials* sweep — the T loop, the MSE-vs-T curves and the RNG are its whole
shape — whereas sparsity is a static property of the tensor: no trials, no
averaging, no recipe at all for the raw numbers. They share the leaf quantizers,
`dsv3_dumps` discovery and the recipe table, and no control flow.

Three numbers that are routinely conflated:

| | what it is |
| --- | --- |
| `exact_zero` | exactly 0 in the dump. A property of the **model**: ReLU families produce them, SiLU/GELU families essentially never do, MoE routing produces them in whole rows. Costs nothing — FP4 represents zero exactly. |
| `flush` | **nonzero in, FP4 code 0 out**. This is information the format destroys, and it is *block relative*: an element flushes when it is small against the amax of its own 16-element block, not when it is small absolutely. |
| `dead_block` | the E4M3 block scale itself underflows, so the whole block reconstructs as zero. |

The distributional statistic behind `flush` is `p50_rel`, the median of
`|x| / block_amax`. An element rounds to FP4 zero at about `|x| < block_amax/24`
(the E2M1 RNE threshold 0.25 against the 6.0 ceiling), so `p50_rel` says how much
headroom a typical element has, and it is recipe-independent — which is what
makes it comparable across models. `test_sparsity.py` checks the measured
`flush` against that analytic threshold by two independent routes.

**The RHT is a sparsity-destroying transform.** It mixes 16 (9004) or 128
(100483) elements per output, so a rotated lane has essentially no exact zeros
and a much tighter `|x|/block_amax` spread. Sparsity is therefore reported per
lane with the rotation that lane actually gets, and the raw pre-rotation
exact-zero fraction is carried alongside so the model property stays visible.

### Running it on real dumps

Discovery, filtering and tensor preparation are the same path
`plot_bias_heatmaps.py` used to produce E21, so the flags carry over:

```bash
python3 -m $RB.analyze_sparsity --base-dir /path/to/dumps --tensor-type G \
    --variants G G.T --recipe 9004 --csv sparsity.csv
# same subsetting flags as plot_bias_heatmaps.py
#   --layer-types ... --skip-layer-numbers ... --exclude-experts
#   --rank N --step N
```

Dumps are handed to the quantizer at their own dtype (both quantizers accept
fp32; nothing is silently rounded to bf16 first) and `flatten_to_2d` collapses a
`[seq, batch, hidden]` dump the same way the sweep does. Ragged MoE expert row
counts are fine: the quantizer pads to `(32,16)` / `(128,64)` and the RHT to a
multiple of its dimension, and the padding is masked out of every statistic —
`test_sparsity.py` pins that with shapes like `1373 x 1408` (`1373 % 128 == 93`)
on both lanes, rotated and not.

Both scripts print flat `name: value` scalars on stdout for a CI metric
scraper, unconditionally — `report_sweep_metrics` in `plot_bias_heatmaps.py`
and `report_sparsity_metrics` here:

```
sparsity_100483_g_flush_median_pct: 6.662868
sparsity_100483_g_dead_block_max_pct: 0.000000
sparsity_100483_g_moe_routed_fc1_flush_median_pct: 6.496546
sparsity_100483_gt_p50_rel_median: 0.340071
```

Recipe and lane are in the *name* because one run sweeps several lanes and a
`stdout_regex` metric cannot associate two lines with each other; group labels
are slugged (`moe/routed/fc1` -> `moe_routed_fc1`, `G.T` -> `gt`). `dead_block`
is reported as the worst rather than the median, because one dead block in one
tensor is a red flag a median over 345 of them would hide. Defining the metrics
in the script rather than in the CI recipe keeps them under the repo's tests and
means a hand run emits exactly what CI reads.

### Exact zeros in the DSV3 dumps are negligible

From the committed E21 summaries (345 G tensors, `n_elements` in the zeros
bucket), median exact-zero fraction by module:

| module | zeros % |
| --- | ---: |
| `qkv` | 0.057 |
| `proj` | 0.041 |
| `fc2` (routed / shared / dense) | 0.037 |
| `fc1`, `fc3`, `kv_a`, `kv_b` | **0.000** |

So structural sparsity is not what matters for these dumps; `flush` is.

### GLU vs non-GLU

`--synthetic` builds fc1/fc2/fc3 gradients for one MLP block with the shapes,
input distribution and init held fixed, so the only thing varying is the
nonlinearity. `G_fc2` is the incoming `dy` and is untouched by the activation,
which makes it the control — its ~6.8% flush is the **Gaussian floor** for 1x16
NVFP4 blocks, and anything above that is excess dynamic range the activation
introduced.

Recipe 9004, `--variant G` (the dgrad lane, which 9004 leaves **unrotated**):

| activation / layer | raw zeros % | flush % | total FP4 zeros % | p50_rel |
| --- | ---: | ---: | ---: | ---: |
| `swiglu/fc1` | 0.0 | **36.1** | 36.1 | 0.087 |
| `swiglu/fc3` | 0.0 | 27.0 | 27.0 | 0.113 |
| `gelu/fc1` | 0.0 | 26.2 | 26.2 | 0.137 |
| `relu2/fc1` | **50.0** | **8.8** | 58.8 | 0.216 |
| any `fc2` (control) | 0.0 | 6.8 | 6.8 | 0.333 |

The naive reading of that table is backwards. `relu2` has the *most* zeros
(58.8%) but discards the *least* information (8.8%), because half its gradient
is exactly zero by construction — `d(relu²)/dh = 2·relu(h)` vanishes on half its
domain — and an exact zero is free. SwiGLU has no exact zeros at all but flushes
36% of fc1, because gating spreads the gradient over a much wider within-block
dynamic range. A non-GLU MLP is therefore **easier** for NVFP4 on this lane, not
harder.

Under recipe 100483 every lane is rotated at dim 128 and the whole table
collapses to 6.44–6.83% flush regardless of activation: RHT-128 Gaussianizes the
operand and erases the difference. The activation choice is a V1-dgrad-lane
concern, not a V2 concern.

## Equivalence: what is proven, and what is not

Proven by `test_nvfp4_qdq.py` (no kitchen needed):

1. **CuTe DSL == PyTorch oracle, bit for bit** — the bf16 QDQ output and every
   intermediate (FP4 codes, E4M3 block scales, global scale) across padded and
   unpadded shapes, both block axes, RNE and SR, and the degenerate cases
   (all-zero tensor, zero blocks, saturating blocks, signed zeros).
2. **The E4M3 block-scale rounding == `tensor.to(torch.float8_e4m3fn)`** —
   checked exhaustively over *every* fp32 bit pattern in `[0, 449)` plus the
   saturating tail.
3. **The oracle == kitchen's own reference math** — `test_reference_matches_kitchen_source`
   re-derives the result from a verbatim transcription of `nvfp_utils` and
   `cast_utils` and compares bits.

Also proven by `test_nvfp4_qdq.py` for the MS-EDEN path: CuTe DSL == oracle bit
for bit on both lanes, padded and unpadded, SR and RNE, plus the degenerate
blocks; the correction is exactly 1.0 on a block already on the FP4 grid and
falls back to 1.0 rather than dividing by zero when every code is zero; the data
codes are identical across seeds while the block scales are not (the structural
difference from 9004); and `emulate_sr_e4m3` is unbiased over 4096 draws.

Proven by `test_kitchen_equivalence.py`, against a real kitchen build
(`KITCHEN_CUDA_ARCHS=100a pip install --no-build-isolation -e .`):

4. **Both implementations == `kitchen.nvfp_utils.to_nvfp` / `from_nvfp`, bit for
   bit**, for RNE and for SR (kitchen's `cast_to_fp4_e2m1_sr` is fed this
   package's Philox uniforms so both sides round on identical random bits), on
   both block axes and on padded and unpadded shapes. A randomized sweep over
   shapes and magnitudes from 1e-4 to 1e4 was bitwise identical in 48/48 cases.
5. **Both implementations == recipe 6302 driven through the real op**, i.e.
   `QuantizeOpNVFP4Emulation.quantize()` / `.dequantize()` (which calls the psx
   kernels), bit for bit, for X and W on both paths. This is the check that the
   padding constants and the amax are right, not just the block arithmetic.
6. **The 9004 rotation == kitchen's, bit for bit** — `rht.transform` forward and
   inverse against `perform_random_hadamard_transform_ref`, including the fused
   `diag(s) @ H` matrices themselves, padded and unpadded shapes. (kitchen folds
   `1/sqrt(16)` into the cuBLAS `alpha`, i.e. it scales the fp32 accumulator
   before rounding, while `rht.transform` scales after; 0.25 is a power of two,
   and this test is what says so rather than the argument.)
7. **The whole 9004 leaf == kitchen's, bit for bit** — rotate, quantize,
   inverse-rotate, crop — for X and X.T through the real
   `QuantizeOpFP8HadamardTransform`, both backends. Which lanes rotate is pinned
   separately against `get_rht_settings_for_tensor`; G's lanes cannot be compared
   bitwise because `use_sr` makes both of them stochastic, so the rotated G.T
   tensor is checked with the SR bracket technique instead.

Proven by `test_kitchen_equivalence.py` for recipe **100483** specifically:

10. **The MS-EDEN leaf == kitchen's Eden kernel, bit for bit, stochastic
    rounding included** — both lanes, three shapes, through
    `QuantizeOpEdenFP4Emulation.quantize()` / `.dequantize()`. This is the claim
    the 9004 SR path can never make, and it needs no monkeypatching: 100483's
    data is RNE and its one random step is `emulate_sr_e4m3`, a **software** bit
    twiddle on a Philox word whose seed is a plain kernel argument, so both sides
    can be made to round on identical bits by pinning `rng_seed` and switching
    off `iterate_rng_seed` / `mix_rank_in_seed`. The deterministic
    (`stochastic_round_scale=False`) branch is checked separately, so a
    correction bug and a compensating RNG bug cannot cancel.
11. **The dim-128 rotation == kitchen's, bit for bit, on both lanes**, forward
    and inverse, including the fused `diag(s) @ H` matrices. The identity
    (dgrad) lane — the `x @ H` form, mixing 128 columns — is new here; 9004 never
    rotated one.
12. **The whole 100483 leaf == kitchen's, bit for bit** — rotate, quantize,
    inverse-rotate, crop — for G and G.T, both backends, through the real
    `perform_random_hadamard_transform_ref` and the real Eden op.
13. **The recipe table is pinned against the config**: which GEMMs rotate and at
    what dimension (`get_rht_settings_for_tensor`, `hadamard_dim_*`), that G's
    two lanes draw different checked-in sign vectors, that `dynamic_signs`
    tracks `enable_online_randomization`, and that G uses the Eden quantizer
    while X and W keep the psx NVFP4 one.

Proven by `test_psx_clippy_equivalence.py`, which needs only a built
`psx_formats` (kitchen's `third_party/psx-formats`):

8. **Both implementations == `psx_formats.utils.nvfp4_clippy` /
   `nvfp4_clippy_transpose` with `scale_rounding_mode=E4M3_RNE`, bit for bit** —
   the CUDA kernels the recipe actually dispatches to — on both paths, on padded
   and unpadded shapes, and over a randomized sweep of shapes, magnitudes 1e-4
   to 1e4 and 10%-zero tensors (24/24 cases, both paths).
9. **Under SR, clippy rounds on the same grid.** Every element of a clippy SR
   reconstruction lands on one of the two FP4 grid points this package's SR
   chooses between (32 trials x 131k elements, both paths), and the trial mean
   converges on the input. That is as close to bitwise as the SR path can get:
   clippy draws from `curand` inside the kernel and rounds by adding random bits
   to the mantissa, so the stream can never be shared.

End to end, kitchen's own `analyze_rank_bias.py` and this port produce **exactly
equal CSVs** (all 32 bucket x trial rows, relative difference 0.0) on the same
dump, with either backend, for recipe 6302 variants X and X.T *and* for recipe
9004 variant X.T — the rotated lane included. For 9004 `G.T` the two SR streams
are independent so exact equality is impossible; the measured curves agree to
2.4% on average (7.7% worst, at T=2 where trial variance is largest) and the
bucket-mean MSE agrees to under 1% at every checkpoint.

Not proven, and worth knowing:

* **Three regimes where clippy and kitchen's own reference math disagree**, all
  pinned by `test_known_divergences_from_clippy`. This package follows the
  reference (and therefore `nvfp_utils`) in each:
  * an input element that is exactly `-0.0` comes back as `-0.0` from clippy
    (`copysignf`) and `+0.0` here (`cast_utils` multiplies by `torch.sign`).
    Same value, different sign bit.
  * `global_amax * block_amax` overflowing fp32 (an amax around 1e19 and up)
    trips clippy's combined-amax guard, which zeroes the block; the reference
    reconstructs it.
  * an E4M3 block scale that underflows to zero — block amax below `2**-9 / 448`
    of the tensor amax — while the block still holds values above the FP4
    rounding threshold: clippy substitutes a unit block scale and returns
    `code * global_decode_scale`; the reference dequantizes through the zero
    scale and returns 0.

  The last two need a tensor with roughly 2e5:1 dynamic range or an amax near
  the fp32 ceiling, so a bf16 gradient dump does not reach them.
* **Nothing about 100483 has been run against a real DSV3 dump.** The
  9004-vs-100483 numbers quoted above are on a Gaussian tensor, because no dump
  is present in the environment this was developed in. The bitwise equivalences
  are dump-independent; the *curves* are not, and E21's finding (a flat amax
  bucket on real gradients where a Gaussian shows slope -0.9) is exactly the
  reason not to generalize from the Gaussian.
* **100483's real sign vectors are re-drawn from kitchen's ambient generator.**
  The `["128"]` vectors checked into `hadamard_random_sign_vec.json` are what
  this port draws when the sign is frozen, and freezing is the only setting that
  reproduces kitchen bitwise — but 100483 itself never freezes, so the bitwise
  tests pin a configuration the recipe does not run in. `--vary-rht-sign` (the
  default for 100483) matches the recipe's *distribution*, never its bits, which
  is the same situation SR is in for 9004.
* **`--variant G` is not comparable across 9004 and 100483.** Under 9004 the
  identity lane is unrotated, so `G` was bit-identical to 6302; under 100483 it
  carries RHT-128. E21's `g_9004` "dgrad, no RHT" column therefore has no 100483
  counterpart, and only `G.T` compares like for like.
* **A re-drawn RHT sign vector cannot be compared bitwise.** `--vary-rht-sign`
  draws from an explicit per-trial generator; kitchen's online randomization
  draws from the ambient one. Same distribution, different stream — the same
  situation SR is in. The default fixed vector *is* bitwise-comparable, and is
  what 9004 itself uses.
* **Stochastic rounding uses a different random stream.** Kitchen draws
  `torch.rand` from the ambient CUDA generator; here every element draws from
  Philox-4x32-10 keyed by `(seed, element index)` — the only way two independent
  implementations can be bitwise equal under SR, and it makes a trial
  reproducible. Same distribution, different stream, so per-element outcomes
  differ from a kitchen run even though the statistics do not.
* **Kitchen's two scale expressions round differently, and both matter.**
  `torch.div(6*448, amax)` (Python scalar *numerator*) is a true IEEE division,
  while `torch.div(block_amax, 6)` (Python scalar *denominator*) multiplies by
  the fp32 reciprocal of the scalar; the `scalar / tensor` operator would be a
  third thing again (reciprocal-multiply, 1 ULP from a true divide on ~25% of
  inputs). Both are spelled out explicitly in `nvfp4_reference.quantize`.
  Getting the first one wrong moved 2 elements in 6240 by one FP4 step.
* **Subchannel (2-level block) scaling, UE5M3 scales, 1x32 tiles and the
  2D-block tile shapes are not modelled.** (EDEN now is, for 1x16 blocks with
  `correction_dim == 16`, which is the configuration 100483 uses; the pooled
  `correction_dim != 16` and E0M3 Eden kernels are not.) That leaves `--variant W` / `W.T`
  unavailable under 9004 (16x16 `PER_2D_BLOCK`), and recipes 6305 and the EDEN
  grid out of scope entirely. The psx 2D clipper is the same two-level NVFP4
  scaling with a 16x16 amax, minus the 1D path's two `isinf` guards, if W is ever
  wanted.
* An all-zero *tensor* takes kitchen's `_zeros_early_return` path, which returns
  `+0.0` everywhere; both implementations here instead fall through the unit-scale
  guard, which preserves `-0.0` for negative-zero inputs. Values are equal, bits
  differ for that one degenerate input.

One inherited quirk: the sweep aggregates squared error per rank bucket with
`Tensor.scatter_add_`, whose atomic ordering is nondeterministic, so the last
digit or two of a reported bucket MSE varies between runs of the *same* backend.

## Performance

4096x4096 bf16, GB200, full QDQ (amax + quantize + dequantize):

| backend | ms / trial |
| --- | --- |
| `cutedsl` | 0.41 |
| `torch` | 5.4 |

The CuTe entry points `cute.compile` once per argument layout and cache; the
per-trial seed is a runtime argument, so seeding a new trial does not recompile.
