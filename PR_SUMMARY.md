# NVFP4 training: CuteDSL kernels, fast-math path, and AUTO backend selection

## Summary

Brings the CuteDSL NVFP4 quantization kernels to parity-or-better with the Triton
backend across the DeepSeek-V3 shapes, adds a fast-math path to **both** backends, and
promotes CuteDSL + fast math to the default via `KernelPreference.AUTO`.

CuteDSL now leads Triton by **1.6x on grouped RTNE, 2.1x on grouped SR, 1.7-1.8x on the
grouped amax, and 1.8x on grouped 2D weights** at 671B, and fast math is worth a further
1.3-1.4x on top. Round-to-nearest-even output stays bitwise identical to both the Triton
backend and the TransformerEngine-derived reference on every path.

Five commits on top of `main`.

## Key details

### 1. Linear CuteDSL SR: murmurhash3 -> Philox

The linear CuteDSL kernels derived stochastic-rounding bits from a murmur3 32-bit
finalizer (`_hash_u32`) — a well-mixed hash, but not the generator Triton or
TransformerEngine use, so linear SR could not be compared against either. Replaced with
Philox-4x32-10, the same generator `triton.language.random` uses.

### 2. Fast-math path on both backends

Fast math matches TransformerEngine under `NVTE_USE_FAST_MATH=1` and does two things:

- **Skips the FP32 -> BF16 -> FP32 round-through of the tcgen05 RHT accumulator.**
  Gated per call site: it applies to the columnwise path only, since the rowwise path has
  no accumulator.
- **Replaces the correctly rounded `div.rn` encode reciprocal with `rcp.approx.ftz.f32`**
  (one `MUFU.RCP`). No `tl` builtin or libdevice entry reaches this instruction —
  `rcp_rn/rd/ru/rz` are correctly rounded and `fast_dividef` is `div.approx`, a different
  instruction — so Triton gets there through `tl.inline_asm_elementwise`. CuteDSL uses
  `cute.arch.rcp_approx`; TransformerEngine uses CUTLASS `reciprocal_approximate_ftz`.
  All three lower to the same SASS.

Triton previously had no fast variant at all. Adding it keeps the two backends bitwise
identical to each other **and** to TE in either mode, so `AUTO` and `TRITON` agree.

Weight quantization is deliberately untouched: without an RHT there is no accumulator to
skip, and TE has no 2D fast path either (`quantize_transpose_nvfp4.cuh` has zero
`use_fast_math` references). The amax kernels are always exact, in TE and CuteDSL both.

### 3. CuteDSL + fast math as the default

`NVFP4TrainingConfig.kernel_preference` moves `TRITON` -> `AUTO` and `use_fast_math`
defaults `True`. `AUTO` resolves to CuteDSL on SM100 with the CuteDSL runtime present and
falls back to Triton otherwise.

Both defaults are documented in the config with the exact settings that recover prior
numerics (`kernel_preference=KernelPreference.TRITON, use_fast_math=False`), so a
loss-curve diff across this branch does not silently carry two moved variables.

Fast-vs-exact measures 30.2-32.6 dB SQNR on the linear path and 30.4-31.9 grouped —
roughly 10 dB quieter than NVFP4's own ~20 dB quantization noise. The error is large and
rare rather than small and uniform (~1.2% of code bytes differ): skipping the bf16
round-through moves a value by ~2**-9, but an element near an E2M1 midpoint then flips a
whole FP4 step.

### 4. Philox divergence between CuteDSL and Triton

**Under RTNE the two backends are bitwise identical. Under SR they are not**, and this is
deliberate.

Triton's SR path calls `tl.randint(seed, offset)` once per packed byte. Two things waste
work there:

- `tl.randint` computes a full Philox-4x32-10 draw (4 words) and returns **one** of them.
- The `cvt.rs` conversion goes through `tl.inline_asm_elementwise` with `pack=4`, and
  `pack` applies uniformly to every argument. So the asm block receives **4** random words
  while its two `cvt.rs.satfinite.e2m1x4.f32` instructions consume only **2** (`$9`,
  `$10`) — the other two are computed and discarded. There is no way to request a
  different pack factor for the random-bit argument than for the data arguments.

CuteDSL is not bound by that shape. `philox4_all` draws **one counter per 16-element
block and consumes all four output words**, computing all four in the last round:
**34 multiplies per block instead of 124**. The counter is derived from tile coordinates
rather than a running per-thread value, because the persistent CLC scheduler's visit
order is not fixed — so results stay reproducible for a given `rng_state`.

The consequence: CuteDSL's SR stream is a *different, equally valid* stream, not a
matching one. It is validated on properties rather than bit-equality — every SR code
within one FP4 magnitude step of the bitwise-checked RTNE code, reconstruction SQNR,
unbiasedness at exact-halfway values, and `rng_state` reproducibility.

This is the one behavioural asymmetry in `AUTO`: on a node without the CuteDSL runtime,
the silent Triton fallback changes backward-pass numerics under SR, not just which kernel
runs. Pin `kernel_preference` explicitly for runs that must reproduce bitwise across
machines.

## Major optimizations

Seven SASS-driven rounds. The method throughout: dump the compiled SASS before spending a
benchmark, find the dominant instruction family, fix that. Three register/occupancy
hypotheses were refuted for free that way, before any code was written.

| # | Change | Effect |
|---|---|---|
| 1 | BF16->FP4 epilogue fusion — scale 8 BF16-origin values with 4 packed `mul.f32x2`, convert straight to 4 packed FP4 bytes | removes the intermediate 16-element FP32 tensor |
| 2 | Vectorized/paired epilogue SMEM reads + fused RHT-accumulator epilogue | 2D weight kernel 3432 -> 2712 static instrs; 128 `LDS.U16` -> 16 `LDS.128` |
| 3 | Grouped SR: fuse multiply+clamp into the convert; `philox4_all` | excess over RTNE +2360 -> +736 static instrs; **~49% faster** |
| 4 | Linear SR: inherits round 3's fusion, plus one-draw-per-block Philox | excess +3872 -> +1112; **~45% faster** |
| 5 | Linear RHT amax vectorization + columnwise SF store packing | amax 2088 -> 1704 static instrs, up to 37.7% faster; SF store 25 -> 9 `STS.U8` |
| 6 | Widen BF16 by shift instead of per-element `PRMT`+widen | grouped row block 455 -> 423 static instrs; linear exact -3.05% |
| 7 | 64-bit grouped row-code stores (8 `STG.E` -> 4 `STG.E.64`) | store sectors 4,587,520 -> 3,670,016; grouped fast -3.05% |

Refuted and recorded so they are not re-derived: `warpgroup_reg_alloc/dealloc` on the
linear amax (`setmaxnreg` redistributes within a CTA's existing allocation, it does not
lower the compiled base); raising occupancy past `NUM_SMS` (monotonically worse at every
shape); an SR register-spill hypothesis (both variants report `REG:128`, no `MOV.SPILL`).

Also fixed along the way: `cutedsl_prepare_for_cuda_graph` never warmed the kernels it
claimed to — `lru_cache` keys on the literal `(args, kwargs)` shape, so the warm-up's
keyword call and the runtime's positional call were two entries compiling the same
kernel. Twelve kernels compiled, zero hits. All parameters are now required positionals so
the two cannot diverge, and a test asserts zero lazy compiles after `prepare()`.

## Performance: CuteDSL vs Triton, DeepSeek-V3 671B

GB200, CUDA 13.4, PyTorch 2.15.0a0+git0f3e7e2, Triton 3.8.0, nvidia-cutlass-dsl 4.5.2.
Device kernel self-time in microseconds, median of three full script passes at 15 warmups
/ 50 timed profiler iterations. `E = 4` local experts, the EP=64 training layout.
Reproducible with:

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col --experts 4 --math all
python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax --experts 4
python -m benchmarks.prototype.nvfp4_training.bench_group_quantize_2d
```

### Grouped RHT quantize — round-to-nearest-even

| projection | math | CuteDSL | Triton | speedup | CuteDSL fast/exact |
|---|---|---:|---:|---:|---:|
| gate/up (w1/w3) | standard | 53.95 | 87.49 | **1.62x** | — |
| gate/up (w1/w3) | fast | 39.73 | 64.16 | **1.62x** | 1.36x |
| down (w2) | standard | 55.23 | 86.91 | **1.57x** | — |
| down (w2) | fast | 41.14 | 63.71 | **1.55x** | 1.34x |

### Grouped RHT quantize — stochastic rounding

| projection | math | CuteDSL | Triton | speedup | CuteDSL fast/exact |
|---|---|---:|---:|---:|---:|
| gate/up (w1/w3) | standard | 74.75 | 159.96 | **2.14x** | — |
| gate/up (w1/w3) | fast | 54.93 | 141.28 | **2.57x** | 1.36x |
| down (w2) | standard | 76.51 | 159.30 | **2.08x** | — |
| down (w2) | fast | 56.51 | 140.86 | **2.49x** | 1.35x |

The SR lead is larger than the RTNE lead precisely because of the Philox change: SR costs
CuteDSL ~1.4x its own RTNE time against Triton's ~1.8x.

### Grouped amax and 2D weights (no fast-math variant — see above)

| kernel | projection | CuteDSL | Triton | speedup |
|---|---|---:|---:|---:|
| grouped RHT amax | gate/up | 22.95 | 39.91 | **1.74x** |
| grouped RHT amax | down | 22.42 | 40.54 | **1.81x** |
| grouped 2D weight | gate/up | 59.76 | 109.31 | **1.83x** |
| grouped 2D weight | down | 60.02 | 107.99 | **1.80x** |

### Fast-math value by backend

| path | CuteDSL fast/exact | Triton fast/exact |
|---|---|---|
| linear quantize | 1.57-1.91x | 1.14-1.22x |
| grouped quantize, RTNE | 1.32-1.38x | 1.28-1.36x |
| grouped quantize, SR | 1.27-1.36x | 1.13-1.14x |

Fast math is worth far more to CuteDSL because it removes a larger share of what that
epilogue does once its other work is fused. Triton gains little under SR, where the Philox
work it still carries dominates what fast math removes.

## Gap to TransformerEngine

Complete pipeline (amax + quantize) versus TE's `split_quantize`, 671B at `E = 4`. TE is
2.19.0.dev0 built from
[NVIDIA/TransformerEngine@172bd93](https://github.com/NVIDIA/TransformerEngine/commit/172bd93773ad6ee4ba44b460b7f10ef42fc89d57).

| projection | math | rounding | CuteDSL | TE | TE ahead |
|---|---|---|---:|---:|---:|
| gate/up | standard | RTNE | 86.22 | 64.03 | 1.35x |
| gate/up | standard | SR | 106.44 | 87.90 | 1.21x |
| gate/up | fast | RTNE | 70.44 | 55.66 | 1.27x |
| gate/up | fast | SR | 86.24 | 77.74 | **1.11x** |
| down | standard | RTNE | 88.35 | 62.59 | 1.41x |
| down | standard | SR | 108.59 | 86.67 | 1.25x |
| down | fast | RTNE | 71.54 | 54.73 | 1.31x |
| down | fast | SR | 87.43 | 76.47 | **1.14x** |

Broken out by kernel, the remaining gap sits in one place. The **grouped amax already
beats TE** (23.34 vs 25.81); the quantize kernel trails 1.49x grouped and 1.51x linear —
nearly identical, as expected from the two families sharing `_quant16_from_amax`. On 2D
weights CuteDSL wins at the op level, because TE pays 6.74 us for `amax_kernel` +
`zero_amax_kernel` that torchao's cheaper weight amax replaces.

Before this branch's SR work, TE led the SR pipeline by 2.06x / 2.12x at standard math;
it now leads by 1.21x / 1.25x.

> **Caveat.** This section is retained from a run predating optimization rounds 6-7, so
> the CuteDSL column is a lower bound. No checked-in script reproduces it — the benchmark
> modules cover CuteDSL-vs-Triton only, and `--shape-set` does not offer the
> `(2048, 7168)` / `(7168, 2048)` pair. Both sides were measured together in that run, so
> the ratios are internally consistent; re-measuring one side alone would be worse. A
> `bench_vs_transformer_engine.py` would close this.

## Testing

`761 passed / 36 skipped` on a single GB200 across
`test/prototype/moe_training/nvfp4_training/` and `test_nvfp4_grouped_mm.py`.
Tensor-parallel and FSDP2+TP suites require `torchrun` and were not run in this pass.
Ruff lint and format clean.

RTNE output is bitwise identical to both oracles — the Triton backend and the TE-derived
PyTorch reference in `nvfp4_reference.py` — on every path, unchanged by any optimization
round here. SR is covered by structural properties instead; see **Key details 4**.

`test_fast_math_matches_transformer_engine` ties the fast path back to the reference
transitively: the PyTorch oracle cannot host fast math (`torch.reciprocal`, `1/x` and
`x.pow(-1)` are all bitwise equal to the correctly rounded reciprocal, and nothing in ATen
or inductor lowers to `MUFU.RCP` — verified over 4M values), so Triton takes the
fast-path oracle role. That works because Triton's `tl.dot` FP32 accumulator is bitwise
identical to TE's tcgen05 UMMA accumulator.
