# CuTeDSL NVFP4 — BLOCKED on toolchain; grouped 1D SR round approved and ready

Self-contained handoff. Written to survive a container restart; nothing here depends on
files outside the repo.

## Current State

- Branch `nvfp4_moe_cutedsl_split`, HEAD `2b1d4bea`. Working tree clean.
- The last optimization round is complete, validated, and committed (4 commits, below).
- The next round (grouped 1D stochastic rounding) is **approved and fully specified**
  below. **Zero code changes have been made.**
- **BLOCKED: the container was rebuilt with a cutlass-dsl the repo does not support.**
  Every CuteDSL path fails — 239 failed / 17 passed across the four CuteDSL test modules
  on an unmodified tree. Not a code regression; see the diagnosis below.
- **Resolution: downgrade the container to `nvidia-cutlass-dsl==4.5.2`** (the pin the
  repo's own README carries), then start at Round 0.

---

# BLOCKER — toolchain mismatch (diagnosed 2026-08-16)

## What changed

The container is a fresh image built today: every dist-package is stamped `2026-08-16
05:54`, torch `01:42`. It is **not** the image the baselines below were measured in
(torch was `2.15.0a0+git0f3e7e2`, now `2.15.0a0+git8f988c9`).

| package | installed | repo requires |
|---|---|---|
| `nvidia-cutlass-dsl` (+ `-libs-base`, `-libs-core`, `-libs-cu12`, `-libs-cu13`) | **4.6.2** | **4.5.2** (`README.md:70`, `README.md:127`) |
| `apache-tvm-ffi` | **0.1.7** | `>=0.1.11` if 4.6.2 is kept (the DSL says so itself) |
| torch | `2.15.0a0+git8f988c9` | — |
| TransformerEngine | `2.19.0.dev0+172bd93` | — |
| CUDA | 13.4 | — |

`torchao/prototype/moe_training/kernels/mxfp8/cute_utils.py:154` independently confirms
the target: *"once TorchAO's minimum supported nvidia-cutlass-dsl is at least 4.6"* — i.e.
4.5.x is the supported baseline and 4.6 is future work.

`pip check` does not catch this: `nvidia-cutlass-dsl` declares no dependency on
`apache-tvm-ffi` at all, so pip considers the (broken) combination consistent.

## Failure 1 — tvm-ffi API drift (linear 2D and 1D paths)

```
TypeError: make_kwargs_wrapper() got an unexpected keyword argument 'map_dataclass_to_tuple'
```

cutlass-dsl 4.6.2 `cutlass/cutlass_dsl/tvm_ffi_provider.py:659` passes
`map_dataclass_to_tuple=` to `tvm_ffi.utils.kwargs_wrapper.make_kwargs_wrapper`. That
parameter does not exist in apache-tvm-ffi 0.1.7. The surrounding `except ImportError`
in the DSL states the requirement outright: *"install apache-tvm-ffi>=0.1.11"* — but the
call raises `TypeError`, not `ImportError`, so the DSL's own guard does not fire and the
error surfaces raw. Hits everything that reaches the `--enable-tvm-ffi` launcher
(102 cases in `test_quantize_2d.py` alone).

## Failure 2 — stricter `cute.copy` alignment verifier (grouped path)

```
MLIRError: 'cute.copy' op S ptr alignment (64 bits) does not meet requirement (128 bits)
  of atom '!cute_nvgpu.atom.universal_copy<i128, 128 b>'
  ... (!cute.memref<i128, smem, align<8>, "1:0">, !cute.memref<i128, rmem, align<32>, "1:0">)
```

Raised at MLIR verification, i.e. before any tvm-ffi call, so this is independent of
Failure 1. The `rmem align<32>` destination is a 16-element bf16 register tensor (32 B) —
that is `rBlk` in the grouped row epilogue, so the offending copy is

```python
# _cutedsl_group_kernels_impl.py:892-895  (landed in 7b6f3c3e, Experiment 9)
cute.autovec_copy(
    cute.local_tile(sA_clean[(None, tok, stage)], (16,), (hb,)),
    rBlk,
)
```

Ruled out by reading source, so this survives the outage and needs no re-measurement:

- `raw_a = smem.allocate_array(cutlass.BFloat16, a_cosize, byte_alignment=128)`
  (`:509`) — the allocation is 128 B aligned.
- `cute.recast_ptr` **preserves** `ptr.alignment` (it passes `ptr.alignment` straight into
  the new `PtrType`), so `swz_ptr` at `:511` is not where the alignment is lost.

So the 8-byte figure comes from the DSL's alignment inference over the *swizzled, sliced*
view (`a_clean_layout` = `tile_to_shape(MN_SW128 atom, (128, 128, stages))` indexed by a
dynamic `tok`), not from the allocation. 4.5.2 did not verify this; 4.6.2 does. Whether
4.6.2 is *right* to reject it is untested — the underlying access is believed 16 B aligned,
but nothing in the tree proves it to the compiler.

## Fix — downgrade (preferred)

Canonical invocation, straight from `README.md:127`. CUDA is 13.4, so `cu13` is the right
libs package:

```bash
pip install apache-tvm-ffi
pip install nvidia-cutlass-dsl==4.5.2 nvidia-cutlass-dsl-libs-base==4.5.2 \
            nvidia-cutlass-dsl-libs-cu13==4.5.2
```

The image also carries `nvidia-cutlass-dsl-libs-core` and `-libs-cu12` at 4.6.2; make sure
no 4.6.2 component is left behind shadowing the 4.5.2 install.

Verify the downgrade took, before anything else:

```bash
pip show nvidia-cutlass-dsl | grep Version   # expect 4.5.2
cd /home/me/nvfp4/third_party/torchao && PYTHONPATH=$PWD python -m pytest -q \
  test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py \
  -k "test_group_rht_correctness and cutedsl"
```

Both failures must be gone. If only Failure 1 clears, the alignment verifier is not
version-gated after all and Failure 2 is a real latent bug in `7b6f3c3e` — say so and stop.

## Fix — forward-port to 4.6.2 (fallback, only if 4.5.2 is unavailable)

Two independent pieces, in this order:

1. `pip install 'apache-tvm-ffi>=0.1.11'` clears Failure 1.
2. Failure 2 needs the grouped row epilogue's SMEM read to prove 16 B alignment to the
   verifier. `cute.recast_ptr` has no `assumed_align` parameter, but `cute.make_ptr` does
   (`assumed_align=`) and `cute.assume(src, divby=)` exists — either could re-establish the
   guarantee. **Confirm the access really is 16 B aligned before asserting it**; a false
   alignment assumption is a silent memory-corruption bug, not a compile error. The safe
   fallback is to narrow the copy (revert `7b6f3c3e`'s vectorization for this one read),
   which costs the 7.7-22% Experiment 9 bought.

Prefer the downgrade. The forward-port is a separate piece of work from the SR round and
should not be bundled with it.

---

# Completed Round (committed, measured in the OLD image)

```
b9525d64 Document CuTeDSL epilogue SMEM optimization results
5758c187 Fuse the CuTeDSL RHT accumulator epilogue
81abdc31 Pair columnwise SMEM reads in the CuTeDSL 2D weight epilogue
7b6f3c3e Vectorize CuTeDSL rowwise SMEM reads
```

Method that worked: dump the compiled SASS *before* spending a benchmark. The 2D weight
kernel used 49 registers with 53% of its 3432 static instructions being address arithmetic
feeding 256 scalar `LDS.U16` — which killed the occupancy hypothesis and pointed at the
SMEM read shape. Three changes landed on four benchmark runs; static instruction count
tracked the wins, 3432 -> 3048 -> 2712.

Full result tables are in `benchmarks/prototype/nvfp4_training/README.md`, section
"Epilogue SMEM-read and packing optimization". Headline: 2D linear 28.9% faster, 2D grouped
30.7%, 1D linear RTNE 15.9%, 1D grouped RTNE 6.6%.

**Every number in this file was measured in the pre-rebuild image.** Treat them as
provisional until the RTNE sentinel is re-confirmed after the downgrade (Round 0).

## Where CuteDSL stands vs TransformerEngine (DSV3 671B, E=4)

GB200, median of 3 samples (15 warmup / 50 timed), device kernel self-time in microseconds.

**2D weight quantize (no RHT).** TE has no grouped 2D kernel, so its column is 4
single-expert `NVFP4Quantizer(with_2d_quantization=True)` calls.

| projection | stage | CuteDSL | TE | winner |
|---|---|---:|---:|---|
| gate/up | amax | 23.47 | 35.38 | ao 1.51x |
| | quantize | 60.09 | 57.06 | TE 1.05x |
| | total | 83.56 | 92.43 | **ao 1.11x** |
| down | amax | 23.24 | 35.25 | ao 1.52x |
| | quantize | 59.93 | 57.03 | TE 1.05x |
| | total | 83.17 | 92.28 | **ao 1.11x** |

2D weights are done — parity on the kernel, a win on the op. Nothing here justifies
further work.

**1D grouped activation quantize (RHT).** CuteDSL = `cutedsl_group_rht_amax` +
`cutedsl_group_rht_quantize_row_col`; TE = `tex.split_quantize`.

| projection | rounding | math | ao amax | ao quant | ao tot | te amax | te quant | te tot | quant | total |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gate/up | RTNE | std | 26.63 | 59.91 | 86.54 | 25.81 | 38.32 | 64.13 | TE 1.56x | TE 1.35x |
| gate/up | SR | std | 26.56 | 154.27 | 180.84 | 26.01 | 61.58 | 87.59 | TE 2.51x | TE 2.06x |
| gate/up | RTNE | fast | 26.36 | 46.58 | 73.12 | 26.21 | 29.55 | 55.79 | TE 1.58x | TE 1.31x |
| gate/up | SR | fast | 26.42 | 120.30 | 146.88 | 26.14 | 51.54 | 77.68 | TE 2.33x | TE 1.89x |
| down | RTNE | std | 26.72 | 61.43 | 88.15 | 25.84 | 36.84 | 62.68 | TE 1.67x | TE 1.41x |
| down | SR | std | 26.57 | 156.94 | 183.51 | 25.97 | 60.48 | 86.44 | TE 2.60x | TE 2.12x |
| down | RTNE | fast | 26.49 | 47.47 | 73.93 | 26.04 | 28.75 | 54.87 | TE 1.65x | TE 1.35x |
| down | SR | fast | 26.56 | 121.37 | 147.92 | 26.08 | 50.63 | 76.71 | TE 2.40x | TE 1.93x |

The amax stage is at parity everywhere, so all remaining 1D gap is in the quantize kernel.
Both figures are hot-cache (`kernel_time_us` does not flush L2), and TE's `split_quantize`
computes its post-RHT amax internally, which is why the stages are broken out rather than
compared as single totals.

---

# NEXT ROUND (approved, not started): grouped 1D stochastic rounding

## Why

SR is the largest remaining gap. The diagnostic is the SR-over-RTNE ratio *within* each
stack: SR costs CuteDSL **2.6x** its own RTNE time but costs TE only **1.6x**. TE's SR
convert stage is actually *cheaper* than its RTNE one — `cvt.rs...e2m1x4` converts 4 floats
per instruction where `cvt.rn...e2m1x2` converts 2 — so TE spends that saving on RNG and
still comes out ahead.

Two independent causes, both found by reading source; no measurement is pending on them.

**Cause 1 — the RNG discards 3 of every 4 Philox words.** `philox4`
(`_cutedsl_kernels_impl.py:397`) calls `philox_c0` four times per 16-element block at
packed-byte counters `p0, p0+1, p0+4, p0+5`, each running the full round schedule to
extract **only the first of four output words**. Counted from `philox_c0:439`: 2 multiplies
in round 2, 4 in each of rounds 3-9, 1 in round 10 = **31 per call, 124 per block**; per
128x128 tile the two epilogues issue 8192 `philox_c0`, about 254k multiplies. TE issues
**one** `generate4` per block and consumes all four words
(`/opt/nvidia/TransformerEngine/transformer_engine/common/util/curanddx.hpp:36-101`;
`common/util/ptx.cuh:914-990` takes `rbits03` and `rbits47` per 8 elements — exactly 4
words per 16): ~40 multiplies, 128 generated bits mapping exactly onto 16 elements x 8
bits, zero waste.

The 4-counter stride exists purely to reproduce Triton's `tl.randint`-per-packed-byte
stream. **The user approved dropping Triton SR bitwise parity for the grouped kernel**, so
this is fixable.

**Cause 2 — multiply and clamp are not fused into the convert.** The SR path still
materializes a 16-entry FP32 register tensor `q` plus 16 `mul.f32` + 16
`min.xorsign.abs.f32` before `_pack16` (`_cutedsl_kernels_impl.py:779-791`) — exactly the
shape RTNE abandoned in `4a6ec376`/`5758c187`. TE fuses the multiply into the same asm
block and applies no clamp, relying on `.satfinite`. torchao's convert stage is already
TE-equivalent, so this is the whole remaining arithmetic gap: 36 instructions per block
where 12 suffice, plus 16 live FP32 registers.

The same `not sr and not fast_math` gate at `_quant16_from_amax:776` means **RTNE
fast-math also still materializes `q`**. The user asked for that to be fixed in the same
round (Round 1b) — it is bitwise-neutral and applies to the 46.58/47.47 us fast-math rows.

## Constraints

- Sentinels: grouped 1D, E=4, per-expert `(2048, 7168)` gate/up and `(7168, 2048)` down.
  Three samples of `bench_utils.kernel_time_us` (15 warmup / 50 timed), median.
  SR baselines **150.9141** / **151.1979** standard, **114.9064** / **115.8348** fast.
  RTNE grouped sentinel **57.56** standard; RTNE fast-math **46.58** / **47.47**.
- Accept at >= ~2% on the sentinel; retain the round only if aggregate improves >= 3% with
  no applicable case regressing > 2%.
- **RTNE must not regress and must stay bitwise identical to both oracles.** `sr` is a
  compile-time cache key (`_compile_group_fused_kernel:955`), so every SR-only change goes
  behind `const_expr(self.sr)`.
- **Determinism is not negotiable even though bitwise-vs-Triton is.** The grouped kernel
  uses a persistent CLC scheduler, so the SR stream must stay a pure function of tile
  coordinates and thread identity, never of visit order — the property
  `_cutedsl_group_kernels_impl.py:574-578` was written to guarantee. TE gets this from its
  grid mapping; we cannot.
- Grouped 1D only for the RNG change. The linear 1D SR path keeps Triton bitwise parity and
  its 5 `test_cutedsl_vs_triton_stochastic_rounding_bitwise` cases untouched.
- Local commits only; no push or history rewrite. One commit per retained round.

## Round 0 — SASS diff, SR vs RTNE (~20 min, no benchmark)

The discriminating measurement, and it is free. Dump both grouped variants:

```python
_compile_group_fused_kernel(0, True, True,  False)   # SR
_compile_group_fused_kernel(0, True, False, False)   # RTNE
```

under `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=<dir>`, then `cuobjdump -res-usage -sass`.
Diff the instruction mix and record:

- `IMAD`/`IMAD.HI`/`LOP3` delta -> true Philox cost against the 124-multiply estimate.
  Sizes Round 2.
- `FMUL` + `FMNMX` delta -> the unfused multiply/clamp. Sizes Round 1.
- `REG`, `STACK`, and any `LDL`/`STL`/`MOV.SPILL` -> **third hypothesis**: `REG_DEALLOC`,
  `REG_COL = 192`, `REG_ROW = 136` (`_cutedsl_group_kernels_impl.py:103-105`) are plain
  module-level ints shared by every variant, and SR adds 16 live FP32 (`q`) + 4 random
  words + carried Philox state on top. If SR spills and RTNE does not, that is a cheap
  independent win (Round 3).

Also re-run the grouped RTNE sentinel. **It should read ~57.56 us. If it does not, the
downgraded image still differs from the one every number above was taken in, and all
baselines must be re-measured before Round 1.**

## Round 1 — Fuse the SR multiply and clamp into the convert (bitwise-neutral)

**File:** `torchao/prototype/moe_training/nvfp4_training/_cutedsl_kernels_impl.py`

Add two primitives beside the RTNE pair they mirror (`_mul_cvt_rn_e2m1x8_f32:160`,
`_mul_cvt_rn_e2m1x8_acc_f32:223`):

- `_mul_cvt_rs_e2m1x8_f32(v0..v7, scale, rb0, rb1)` — the body of
  `_mul_cvt_rn_e2m1x8_f32` (4x `mov.b64` pack, 4x `mul.f32x2`, 4x `mov.b64` unpack) with
  the four `cvt.rn.satfinite.e2m1x2.f32` replaced by
  `cvt.rs.satfinite.e2m1x4.f32 h0, {a3, a2, a1, a0}, $rb0` and
  `... h1, {a7, a6, a5, a4}, $rb1`, then `mov.b32 $0, {h0, h1}`. TE's exact shape
  (`ptx.cuh:940-990`).
- `_mul_cvt_rs_e2m1x8_acc_f32(...)` — same, with the exact-mode bfloat16 round-through
  (4x `cvt.rn.bf16x2.f32` + shift/mask re-widen) folded in as `_mul_cvt_rn_e2m1x8_acc_f32`
  does, since the grouped **column** epilogue runs `rht_acc=True`
  (`_cutedsl_group_kernels_impl.py:788`).

Nibble order is pinned by existing code and must be preserved: `_pack16:646` calls
`_cvt_rs_..._pack4(q0,q2,q4,q6, q1,q3,q5,q7, rb0, rb1)`, and that asm's `{$6,$2,$5,$1}`
lane order yields nibbles `[q0,q1,q2,q3]` for `h0`. With natural-order arguments the new
primitive expresses the same thing as `{v3,v2,v1,v0}` / `{v7,v6,v5,v4}`. `rb0` covers
`vals[0:4]`, `rb1` `vals[4:8]`, `rb2` `vals[8:12]`, `rb3` `vals[12:16]`.

Add `_pack16_rs_from_enc(vals, enc, rb, use_acc)` shaped like `_pack16_rn_from_enc:739`
(`w0` from `vals[0:8]` with `rb[0], rb[1]`; `w1` from `vals[8:16]` with `rb[2], rb[3]`), and
route the `sr` branch of `_quant16_from_amax` through it so `q` is never materialized.

**Clamp decision, resolved by test rather than argument.** Implement without the explicit
`min.xorsign.abs` first, matching TE and matching what Experiment 12 already validated for
RTNE. The only inputs it can affect are `|x| > 6`, where `.satfinite` also yields +-6 — but
`cvt.rs` perturbs the mantissa before saturating, so this is to be verified, not assumed.
Discriminator (~1 min); this is the **linear** path, which retains full Triton SR parity, so
it is a real oracle here:

```bash
pytest -q test/prototype/moe_training/nvfp4_training/test_hadamard_quantize_row_col.py \
  -k test_cutedsl_vs_triton_stochastic_rounding_bitwise
```

If it fails, add 8x `min.xorsign.abs.f32` inside the asm block and re-run. Either way this
round is bitwise-neutral for both linear and grouped, and lands independently of Round 2.

## Round 1b — Fuse the RTNE fast-math path (bitwise-neutral)

Same file, same function. Fixing the `not sr and not fast_math` gate collapses
`_quant16_from_amax` to one uniform shape:

```python
enc, pvscale_fp8 = _enc_from_amax(amax, enc_over_fp4max, dec, fast_math)
use_acc = rht_acc and not fast_math   # fast math consumes FP32 directly, no bf16 round-through
if cutlass.const_expr(sr):
    w0, w1 = _pack16_rs_from_enc(vals, enc, rb, use_acc)
else:
    w0, w1 = _pack16_rn_from_enc(vals, enc, use_acc)
return w0, w1, pvscale_fp8
```

`use_acc` is the load-bearing detail: for `fast_math and rht_acc` (the grouped column
epilogue in fast mode) the correct primitive is the **plain** one, not `_acc`, because fast
math deliberately skips the bfloat16 round-through.

**Verified cleanup caused by this change.** Once Rounds 1 and 1b are both retained these
become unreachable — confirmed by grep across `torchao/prototype/moe_training`, `test/`, and
`benchmarks/`, which finds no other caller — and should be deleted in the 1b commit:
`_pack16:646`, `_cvt_rs_satfinite_e2m1x4_f32_pack4:92`,
`_cvt_rn_satfinite_e2m1x2_f32_pack4:46`, `_cvt_rn_bf16x2_f32:142`, `_u32_as_f32:323`,
`_min_xorsign_abs_f32:500`. If 1b is reverted on a benchmark result, keep them — the RTNE
fast path still needs `_pack16`.

Gate: the full RTNE two-oracle suite, plus `test_cutedsl_group_fast_math_matches_transformer_engine`
(the byte-identity check against real TE fast math), plus the two RTNE fast-math sentinels.
Separate commit from Round 1 so the SR and RTNE wins stay independently attributable.

## Round 2 — One Philox per block, all four words consumed

**Files:** `_cutedsl_kernels_impl.py` (shared generator), `_cutedsl_group_kernels_impl.py`
(call sites).

Add `philox4_all(state, chunk_counter)` next to `philox4:397`, reusing the **existing**
`philox_prep:370` state unchanged — that hoist is an advantage over TE, which recomputes its
key schedule in-loop, and the counter still enters at the same place (`c1`, with
`c0 = offset_base`, `c2 = c3 = 0`). The only change from `philox_c0:439` is that round 10
computes all four words instead of dropping `c1`/`c3`, and all four are returned.

Cost: 31 -> **34 multiplies**, yielding 4 words instead of 1 — **124 -> 34 per 16-element
block, a 3.6x reduction**, better than a naive port of TE's ~40 because the launch-uniform
prep is retained.

**Counter derivation — the part that must not be copied from TE.** TE walks a per-thread
counter and relies on `blockIdx` for a stable subsequence
(`quantize_transpose_nvfp4_tuned_1D.cuh:383`); our persistent CLC scheduler makes visit
order unstable. Derive from coordinates instead — one counter per 16-element chunk, which is
exactly the current packed-byte expression divided by 8 (`TILE_PACKED = 8192` packed bytes
per tile, 8 packed bytes per chunk; all three terms divide exactly):

- column epilogue (`_cutedsl_group_kernels_impl.py:778-783`): `tile_id * 1024 + h_local * 8 + u`
- row epilogue (`:902-908`): `tile_id * 1024 + tok * 8 + hb`

`tile_id = tile_n * tiles_in_h + tile_m` is already a named variable in the column epilogue
(`:758`); the row epilogue inlines the same product at `:904` and should bind the name. All
operands (`h_local:729`, `u:772`, `tok:891`, `hb:856`, `tiles_in_h:578`) are in scope.
Introduce `TILE_BLOCKS = TILE_PACKED // 8` beside `TILE_PACKED:1657` rather than open-coding
1024.

Column and row keep distinct keys (`sr_rng_t[0..2]` vs `sr_rng_t[4..6]`, as today) so the
streams stay uncorrelated where counters coincide.

This preserves the two properties that matter — same `rng_state` gives identical codes, and
the stream is independent of scheduling. The **linear** kernel
(`_cutedsl_kernels_impl.py:1448`, `:1595`) keeps calling `philox4` with `triton_tile_id`;
both stay.

### Test changes this round requires

`test_cutedsl_group_quantize_matches_triton_bitwise` is parametrized `ids=["rtne", "rs"]`
over 4 module-scoped fixtures (`seed223..seed226`), so the 4 `[seed*-rs]` cases assert
exactly the property being dropped. **Narrow the parametrize to RTNE only** and replace the
SR coverage with statistical guards — per the user's direction that bitwise equivalence is
the wrong contract for a stochastic kernel, and that SR should be judged on recovering the
FP32 value in expectation plus a reconstruction SNR at the same threshold the equivalent
Triton test uses.

**File:** `test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py`

1. **Reconstruction SQNR at the RTNE threshold.** A new test parametrized over `_KERNELS`
   that runs `_run_sr:699` and hands the outputs straight to the existing
   `triton_group_rht_quantize_row_col_ref:273`. That helper already asserts
   `compute_error(...) >= 20.0` dB for both the columnwise (vs post-RHT) and rowwise (vs raw
   A) halves at `:310` and `:322`, and its scale assertions are rounding-mode independent
   (block scales come from the amax, not the codes). Same threshold as the RTNE correctness
   test, applied to both backends, with no new reference code.
2. **Unbiasedness.** Port `test_cutedsl_rht_quantize_sr_unbiased`
   (`test_hadamard_quantize_row_col.py:918`) to grouped: fill A with 1.25 (exactly halfway
   between FP4 grid points 1.0 and 1.5), anchor `A[:, ::16] = 6.0` to pin the block amax,
   use an identity global scale, average K=32 SR draws, and assert the mean converges to
   1.25 with a ~50/50 grid split and no off-grid values. This catches a degenerate or
   correlated stream — the failure mode SQNR alone cannot see.
3. **Determinism** stays covered by
   `test_group_rht_rng_state_controls_stochastic_rounding:721`, which is backend-agnostic
   and never compares against Triton. It must keep passing unchanged; it is the real
   regression guard for the new counter derivation.

Porting deltas, all confirmed against the two modules:

- **Return order.** Linear `_quantize_row_col` returns `(col, col_sf, row, row_sf)`; the
  grouped op returns `(qa, sfa, qd, sfd)` — **row first**.
- **RNG interface.** Grouped takes one 4-element int64 `rng_state`
  `[col_seed, col_offset, row_seed, row_offset]` via `_make_rng_state:212`, not four
  separate `*_base` kwargs. Sweeping the offset across draws means varying elements 1 and 3.
- **amax shape.** Grouped requires a 1-D contiguous `(num_groups,)` float32 tensor, not the
  0-dim scalar the linear test passes.
- **Scale de-swizzle.** Rowwise `sfa` is whole-extent `to_blocked` (use `from_blocked`);
  columnwise `sfd` is blocked **per group** and needs `_from_blocked_grouped:185`.
- **Imports.** The grouped module imports only `assert_codes_bitwise` /
  `assert_scales_bitwise` from `_assertions`; the unbiasedness port needs
  `_dequantize_plain:152` and `from_blocked`, both already available.

### Docstrings and docs to correct

SR-specific interchangeability claims become false for the grouped path. RTNE
interchangeability is unaffected and must still be stated.

- `test_group_rht_quantize_row_col.py:667-671` (the parity test's docstring) and the module
  header's SR bullet list at `:15-18`.
- `benchmarks/prototype/nvfp4_training/README.md:234-236` — the grouped-section claim
  "bitwise identical output — codes and scale factors, RTNE and stochastic rounding alike".
  `:158-160` is the **linear** section and stays true. `:558-561` ("45 cases, including
  stochastic rounding") needs its count and wording updated.
- `_cutedsl_kernels_impl.py:354-357` (Philox constants comment) and `philox4:401-404` —
  scope the "byte-identical to `triton.language.random`" claim to the linear path.
- `_cutedsl_group_kernels_impl.py:574-578` — the expression `tile_n * tiles_in_h + tile_m`
  stays, but its stated *reason* changes from reproducing Triton's tile index to
  guaranteeing order-independence of the SR stream under the CLC scheduler.
- `group_rht_quantize_row_col_cutedsl.py:7-13` — "same output contract" needs an SR caveat.

## Round 3 — SR-specific register budget (conditional on Round 0)

Only if Round 0 shows SR spilling where RTNE does not. `REG_DEALLOC = 32`, `REG_COL = 192`,
`REG_ROW = 136` (`_cutedsl_group_kernels_impl.py:103-105`) are module-level ints used by
both the fused kernel (`:582, :624, :645, :649, :701, :854`) and the amax kernel; the
comment records the budget as `128*32 + 128*192 + 256*136 = 63488 <= 65536` — about 2048
registers of headroom, 8 more per row thread. Since `sr` is already a compile-time cache
key, making these hints `sr`-dependent costs the RTNE variant nothing. Rounds 1 and 2 both
remove SR-live registers, so re-read the SASS after them before deciding.

## Verification

Sentinel driver (swap shape for `down`, `False` for the RTNE rows). `PYTHONPATH=$PWD`
matters: an older `torchao` in `/usr/local/lib/python3.12/dist-packages` shadows the repo
whenever cwd is not the repo root, and it is missing newer submodules.

```bash
cd /home/me/nvfp4/third_party/torchao && PYTHONPATH=$PWD python - <<'PY'
import statistics, torch
from benchmarks.prototype.nvfp4_training.bench_utils import kernel_time_us
from torchao.prototype.moe_training.nvfp4_training.group_rht_quantize_row_col_cutedsl import (
    cutedsl_group_rht_quantize_row_col,
)
from torchao.prototype.moe_training.nvfp4_training._cutedsl_kernels_impl import DEFAULT_SIGN_VECTOR
torch.manual_seed(123)
E, M, N = 4, 2048, 7168
P = E * M
A = torch.randn(P, N, dtype=torch.bfloat16, device="cuda")
offsets = torch.arange(M, P + 1, M, dtype=torch.int32, device="cuda")
lpl = torch.tensor([P], dtype=torch.int32, device="cuda")
amax = A.view(E, M, N).float().abs().amax(dim=(1, 2)).contiguous()
rng = torch.randint(-(2**63), 2**63 - 1, (4,), dtype=torch.int64, device="cuda")
run = lambda: cutedsl_group_rht_quantize_row_col(
    A, DEFAULT_SIGN_VECTOR, offsets, E, P, N, 0, amax, amax, rng, True, lpl, False)
s = [kernel_time_us(run) for _ in range(3)]
print(f"grouped SR E={E} {M}x{N} median={statistics.median(s):.4f} us")
PY
```

**SR gate** (18 cases before the Round 2 test rewrite; substitute the two new grouped SR
test names for the `-rs` clause afterwards):

```bash
pytest -q test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py \
          test/prototype/moe_training/nvfp4_training/test_hadamard_quantize_row_col.py \
  -k "(test_cutedsl_group_quantize_matches_triton_bitwise and rs) or test_cutedsl_vs_triton_stochastic_rounding_bitwise or test_cutedsl_rht_quantize_sr_unbiased or (test_group_rht_rng_state_controls_stochastic_rounding and cutedsl) or (test_group_rht_stochastic_rounding_launches and cutedsl)"
```

**RTNE non-regression** — the two-oracle suite must still pass unchanged (45 Triton-parity
+ 54 TE-reference), and the grouped RTNE sentinel must stay at 57.56 us:

```bash
# oracle 1 (Triton bitwise)
pytest -q test/prototype/moe_training/nvfp4_training/{test_quantize_2d,test_hadamard_quantize_row_col,test_group_rht_quantize_row_col,test_group_quantize_2d}.py \
  -k "test_cutedsl_weight_quantize_2d_matches_triton or test_cutedsl_vs_triton_interchangeable or test_cutedsl_group_quantize_matches_triton_bitwise or test_cutedsl_group_quantize_2d_matches_triton or test_cutedsl_vs_triton_stochastic_rounding_bitwise"
# oracle 2 (TE reference)
pytest -q test/prototype/moe_training/nvfp4_training/{test_quantize_2d,test_hadamard_quantize_row_col,test_group_rht_quantize_row_col,test_group_quantize_2d}.py \
  -k "(vs_transformer_engine_reference or test_group_rht_correctness or test_group_rht_deepseek_dimensions_correctness) and cutedsl"
# TE fast-math byte identity (matters specifically for Round 1b)
pytest -q test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py \
  -k test_cutedsl_group_fast_math_matches_transformer_engine
```

Plus `ruff check` and `git diff --check`. Finally re-run the E=4 CuteDSL-vs-TE comparison,
update the benchmark README tables, and append each experiment to `debug-session.md` in its
existing hypothesis/evidence/action/result/classification format.

## Restart Checklist

1. Confirm the GPU: `nvidia-smi -L` and
   `python -c "import torch; print(torch.cuda.get_device_name(0))"`.
2. **Confirm the downgrade:** `pip show nvidia-cutlass-dsl | grep Version` -> `4.5.2`.
3. **Confirm CuteDSL works at all** — this is the check the whole session was blocked on:
   `PYTHONPATH=$PWD python -m pytest -q test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py -k "test_group_rht_correctness and cutedsl"`.
4. Re-establish the RTNE baseline before changing anything — grouped RTNE sentinel should
   read ~57.56 us. If it does not, the image still differs from the one all the numbers
   above were taken in, and the baselines must be re-measured.
5. Start at Round 0. No code changes are pending.

## Confidence / Risk

- Confidence **HIGH** that the blocker is the toolchain, not the code: the tree is clean at
  a commit whose tests passed, the repo pins 4.5.2 while 4.6.2 is installed, and one of the
  two failures is cutlass-dsl 4.6.2 calling a tvm-ffi API that its own error message says
  needs a newer apache-tvm-ffi than the image ships.
- Confidence **MEDIUM-HIGH** that the downgrade alone clears both failures. Failure 1 is
  certain. Failure 2 is inferred from 4.6.2 having a stricter `cute.copy` verifier; it has
  not been observed passing on 4.5.2 *in this image*.
- Confidence **HIGH** in the SR diagnosis: both causes come from reading torchao and TE
  source, with instruction counts derived by hand.
- Confidence **MEDIUM** in the size of the win. Cutting RNG multiplies 124 -> 34 and
  arithmetic 36 -> 12 per block should move a kernel where SR-specific work is about 60% of
  runtime, but the SASS diff in Round 0 is what turns that into a number.
- Risk **MEDIUM**. Round 2 deliberately changes observable output for SR and rewrites four
  test cases; it is the first change in this project that is not bitwise-preserving.
  Rounds 1, 1b and 3 are bitwise-neutral and independently revertible.

## Environment

NVIDIA GB200 x4, CUDA 13.4, driver present and healthy. Target toolchain:
`nvidia-cutlass-dsl==4.5.2` (+ `-libs-base`, `-libs-cu13`) with `apache-tvm-ffi`, per
`README.md:127`. Currently installed: 4.6.2 with apache-tvm-ffi 0.1.7 — **broken, see
Blocker above**. torch `2.15.0a0+git8f988c9`, TransformerEngine `2.19.0.dev0+172bd93`.

SASS dump recipe: `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=<dir>` around the compile call,
then `cuobjdump -res-usage -sass <dir>/*.cubin`.
