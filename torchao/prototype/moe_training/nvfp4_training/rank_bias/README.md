# NVFP4 QDQ in CuTe DSL + a standalone `analyze_rank_bias`

A dependency-light port of kitchen's
`experimental/tensor_dump_analysis/analyze_rank_bias.py`. The leaf
quantize/dequantize path is reimplemented twice:

| module | what it is |
| --- | --- |
| `nvfp4_reference.py` | PyTorch oracle, a transcription of kitchen's `nvfp_utils.to_nvfp_verbose` / `from_nvfp_verbose` / `cast_utils` |
| `rht.py` | the wgrad Random Hadamard Transform recipe 9004 adds, a transcription of `perform_random_hadamard_transform_ref` |
| `nvfp4_cutedsl.py` | CuTe DSL kernels (`nvidia-cutlass-dsl`), **bitwise identical** to the oracle |
| `analyze_rank_bias.py` | the sweep, plots and CSV, driving either backend |
| `test_nvfp4_qdq.py` | the equivalence tests |
| `test_kitchen_equivalence.py` | optional cross-check against a *built* kitchen; skipped otherwise |
| `test_psx_clippy_equivalence.py` | optional cross-check against the psx `clippy` CUDA kernels; needs only `psx_formats` |

Nothing imports kitchen, `psx_formats`, TransformerEngine or the rest of the
monorepo. `pip install -r requirements.txt` on any CUDA box (Hopper or newer)
and run.

```bash
pip install -r requirements.txt
pytest -q   # test_kitchen_equivalence.py skips unless kitchen is importable

# real dump
python analyze_rank_bias.py --tensor-path G_rank0_layer0_fc1_step0.pt \
    --recipes 9004 --variant G.T --trials 2 4 8 16 32 64 128 --save-plot-dir ./plots

# both recipes as two curves per bucket
python analyze_rank_bias.py --random 4096x4096 --recipes 6302 9004 --variant G.T \
    --trials 2 4 8 16 --save-plot-dir ./plots
```

## What is modelled

| `--recipes` | SR on G | W tiles | RHT | variants |
| --- | --- | --- | --- | --- |
| `6302` | yes | 1x16 | none | all six |
| `9004` (alias `6304`) | yes | 16x16 2D | wgrad | X, X.T, G, G.T |
| `nvfp4` | no | 1x16 | none | all six |

`--variant W` / `W.T` under 9004 is rejected rather than silently swept with the
wrong tile shape: the 16x16 `PER_2D_BLOCK` path is not implemented. It is only
W that needs it, so the G analysis is unaffected.

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
* **EDEN, subchannel (2-level block) scaling, UE5M3 scales, 1x32 tiles and the
  2D-block tile shapes are not modelled.** That leaves `--variant W` / `W.T`
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
