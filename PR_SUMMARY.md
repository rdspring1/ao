# [nvfp4_training] Optimized CuteDSL kernels for NVFP4 training

## Summary

Brings the CuteDSL NVFP4 quantization kernels to parity-or-better with the Triton
backend across the DeepSeek-V3 shapes, adds a fast-math path to **both** backends, and
promotes CuteDSL + fast math to the default via `KernelPreference.AUTO`.

CuteDSL now leads Triton by **1.6x on grouped RTNE, 2.1x on grouped SR, 1.7-1.8x on the
grouped amax, and 1.8x on grouped 2D weights** at 671B, and fast math is worth a further
1.3-1.4x on top. Round-to-nearest-even output stays bitwise identical to both the Triton
backend and the TransformerEngine-derived reference on every path.

The latest configuration is NVFP4 for all dense linears, shared and routed experts and
MXFP8 attention with fast math enabled. In E2E run with 64 GB300 GPUs for 1 hour, NVFP4
is 37.3% faster than BF16 and +12.8% faster than MXFP8 (1004.6 vs 731.6 and 890.8 TFLOP/s)
while using less memory than either (217.08 GiB, −8.38 GiB vs bf16 and −5.41 GiB vs mxfp8). The
training loss curve is inline with BF16 and MXFP8.

### Key Changes

1. Replace random bit generation in Linear CuteDSL kernels from MurmurHash3 to Philox
2. Add fast-math path to Triton and CuteDSL
3. Create CuteDSL versions of the grouped RHT quantize, grouped amax, and grouped 2D weight kernels
4. Enable CuteDSL + fast math as the default

## Key details

### Fast Math Path

Fast math matches TransformerEngine under `NVTE_USE_FAST_MATH=1` and does two things:

- **Skips the FP32 -> BF16 -> FP32 round-through of the tcgen05 RHT accumulator.**
  Gated per call site: it applies to the columnwise path only, since the rowwise path has
  no accumulator.
- **Replaces the correctly rounded `div.rn` encode reciprocal with `rcp.approx.ftz.f32`**
  (one `MUFU.RCP`).

Triton previously had no fast variant at all. Adding it keeps the two backends bitwise
identical to each other **and** to TE in either mode, so `AUTO` and `TRITON` agree.

Weight quantization is deliberately untouched: without an RHT there is no accumulator to
skip, and TE has no 2D fast path either (`quantize_transpose_nvfp4.cuh` has zero
`use_fast_math` references). The amax kernels are always exact, in TE and CuteDSL both.

### KernelPreference.CuteDSL and Fast Math as the default

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

### Philox divergence between CuteDSL and Triton

**Triton and CuteDSL are bitwise identical for RTNE but not for SR**

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

### Performance: CuteDSL vs Triton

GB200, CUDA 13.4, PyTorch 2.15.0a0+git0f3e7e2, Triton 3.8.0, nvidia-cutlass-dsl 4.5.2.
Device kernel self-time in microseconds, median of three full script passes at 15 warmups
/ 50 timed profiler iterations. `E = 4` local experts, the EP=64 training layout.

Reproducible with:

```bash
python -m benchmarks.prototype.nvfp4_training.bench_group_rht_quantize_row_col --experts 4 --math all
python -m benchmarks.prototype.nvfp4_training.bench_group_hadamard_amax --experts 4
python -m benchmarks.prototype.nvfp4_training.bench_group_quantize_2d
```

<details>
<summary><strong>DeepSeek-V3 671B with 4 local experts</strong></summary>

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
| grouped quantize, RTNE | 1.32-1.38x | 1.28-1.36x |
| grouped quantize, SR | 1.27-1.36x | 1.13-1.14x |

Under RTNE both backends gain about the same. The split appears under SR, where Triton
gains only 1.13-1.14x: the Philox work it still carries dominates what fast math removes.

</details>
