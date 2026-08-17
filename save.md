# CuTeDSL NVFP4 — grouped SR round complete; next gap is grouped RTNE

Self-contained handoff. The toolchain blocker the previous handoff was stuck on is gone.

## Current State

- Branch `nvfp4_moe_cutedsl_split`, HEAD `0b41f58f` plus a docs commit. Working tree clean.
- The approved grouped stochastic-rounding round is **complete, validated and committed**.
- **Grouped SR is ~2x faster**: standard `150.73 -> 77.33 us`, fast math `116.45 -> 57.77 us`.
- RTNE fast math also improved 6.7% (`45.60 -> 42.55`); RTNE standard is unchanged and
  still bitwise identical to both oracles.
- Full `test/prototype/moe_training/nvfp4_training/` suite: **594 passed, 36 skipped**.

### The blocker from the previous handoff is resolved

The container was rebuilt again, back to the image every baseline was measured in:
`nvidia-cutlass-dsl` is **4.5.2** (the README pin) and torch is back to
`2.15.0a0+git0f3e7e2`. All four sentinels reproduced their pre-rebuild baselines within
1.3%, so no re-measurement was needed. Nothing in this file depends on that history.

Note for a future rebuild: `nvidia-cutlass-dsl-libs-{core,cu12,cu13}` are at **4.7.0**
while `nvidia-cutlass-dsl` and `-libs-base` are at 4.5.2. That mix works. `ruff` is not
in the image; install with `pip install ruff==0.11.6` per `CONTRIBUTING.md:18`.

## What Changed

Four commits, one per retained round:

```
64d381cd Fuse the CuTeDSL stochastic-rounding multiply and clamp into the convert
87cb89c1 Fuse the CuTeDSL RTNE fast-math path and drop the scalar quantize
0b41f58f Draw one Philox counter per block in the grouped stochastic-rounding path
<docs>   Record the grouped SR optimization round
```

**Method that worked again: dump the SASS before spending a benchmark.** Diffing the SR
and RTNE compilations of the grouped kernel sized all three candidate causes for free and
killed one of them outright:

| | SR before | SR after | RTNE | excess before -> after |
|---|---:|---:|---:|---|
| total static instructions | 5816 | 4192 | 3456 | +2360 -> +736 |
| `IMAD` | 1511 | 813 | 467 | +1044 -> +346 |
| `LOP3` | 986 | 434 | 228 | +758 -> +206 |
| `FMUL` | 220 | 28 | 28 | +192 -> **0** |
| `FMNMX` | 232 | 40 | 40 | +192 -> **0** |

Both variants reported `REG:128` with no `MOV.SPILL`, which refuted the planned
register-budget round (Round 3) before any code was written. It was dropped.

1. **Fused SR multiply/clamp** — `_mul_cvt_rs_e2m1x8_f32` and `_acc` twin, mirroring the
   RTNE pair. Clamp dropped, relying on `.satfinite`; verified rather than assumed
   against the linear path's 5 triton SR bitwise-parity cases.
2. **Fused RTNE fast math** — collapsed the `not sr and not fast_math` gate to one shape,
   deleting six now-unreachable helpers (186 lines).
3. **One Philox draw per block** — `philox4_all` keeps the `philox_prep` hoist and
   computes all four words in the last round: 34 multiplies against 124. Counter derived
   from tile coordinates, not a running counter, because the CLC scheduler makes visit
   order unstable.

### The one deliberate behaviour change

Change 3 is **the first non-bitwise-preserving change in this project**. Grouped SR codes
are now a different, equally valid stream. Guarded instead by:

- `test_group_rht_sr_reconstructs` — SQNR through the same reference RTNE uses.
- `test_group_rht_sr_unbiased` — ported from the linear test; catches a degenerate or
  position-correlated stream, which SQNR cannot see.
- `test_group_rht_rng_state_controls_stochastic_rounding` — unchanged, backend-agnostic,
  the real guard on the new counter derivation.

**Gap found while writing those tests, worth remembering:** the plan called for SR to
clear the RTNE 20 dB SQNR bar. It does not — it measures 17.2 dB, on **both** backends and
all four fixtures within 0.2 dB. The unmodified triton backend measures the same, so this
is the ~3 dB variance cost of unbiased rounding (error uniform over the quantization
interval rather than bounded by half of it), not a defect. The bar was parameterized
(`sqnr_floor`) and set to 15 dB for SR.

## Where CuteDSL now stands vs TransformerEngine (DSV3 671B, E=4)

Complete pipeline, GB200, median of 3 samples (15 warmup / 50 timed), TE 2.19.0.dev0.
Both sides re-measured at `0b41f58f`.

| projection | math | rounding | CuteDSL | TE | TE speedup | was |
|---|---|---|---:|---:|---:|---:|
| gate/up | standard | RTNE | 86.35 | 64.01 | 1.35x | 1.35x |
| gate/up | standard | SR | 106.23 | 87.92 | **1.21x** | 2.06x |
| gate/up | fast | RTNE | 70.57 | 55.62 | 1.27x | 1.31x |
| gate/up | fast | SR | 86.24 | 77.68 | **1.11x** | 1.89x |
| down | standard | RTNE | 88.51 | 62.65 | 1.41x | 1.41x |
| down | standard | SR | 109.10 | 86.76 | **1.26x** | 2.12x |
| down | fast | RTNE | 71.63 | 54.69 | 1.31x | 1.35x |
| down | fast | SR | 87.34 | 76.43 | **1.14x** | 1.93x |

SR now costs CuteDSL **1.23x** its own RTNE pipeline against TE's **1.37x** — the SR path
is now the more efficient of the two relative to its own baseline.

## What Should Happen Next

**The remaining gap is no longer SR-specific. It is the grouped RTNE quantize kernel.**

The amax stage is at parity with TE everywhere (~23.3 us vs ~25.9), so the whole 1D gap
sits in `cutedsl_group_rht_quantize_row_col`: 57.06 us vs TE's ~38 at standard math for
gate/up. Two facts to start from, both already established:

- After this round, SR runs only 736 static instructions above RTNE, so the RTNE kernel
  itself is now the floor for both. Optimizing it improves all four rows at once.
- The last three rounds all came from the same method — dump the SASS, find the dominant
  instruction family, fix that — and it has not been applied to the grouped RTNE variant
  on its own. The Round 0 dump exists at `/tmp/rtne.sass` (regenerate with the recipe
  below); nobody has yet asked what its 3456 instructions are actually doing.

Concrete next action: dump `_compile_group_fused_kernel(0, True, False, False)`, classify
its instruction mix by family the way Experiment 14 did, and compare against TE's
`quantize_transpose_nvfp4_tuned_1D` SASS for the same shape. Expected outcome: one
dominant family (address arithmetic or SMEM traffic, on the 2D precedent) that accounts
for a third or more of the kernel.

## Constraints That Still Hold

- **RTNE must stay bitwise identical to both oracles.** `sr` is a compile-time cache key
  (`_compile_group_fused_kernel:955`), so SR-only changes go behind `const_expr(self.sr)`.
- **Determinism is not negotiable** even though grouped bitwise-vs-triton no longer is.
  The SR stream must stay a pure function of tile coordinates and thread identity.
- **The linear 1D SR path keeps full triton bitwise parity** and its 5
  `test_cutedsl_vs_triton_stochastic_rounding_bitwise` cases. It still uses `philox4`.
- Accept a round at >= ~2% on the sentinel; retain only if aggregate improves >= 3% with
  no applicable case regressing > 2%.
- Local commits only; no push. One commit per retained round.

## Verification

Sentinel driver — `PYTHONPATH=$PWD` matters, an older `torchao` in
`/usr/local/lib/python3.12/dist-packages` shadows the repo whenever cwd is not the root.

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
for sr in (False, True):
    for fm in (False, True):
        run = lambda: cutedsl_group_rht_quantize_row_col(
            A, DEFAULT_SIGN_VECTOR, offsets, E, P, N, 0, amax, amax, rng, sr, lpl, fm)
        print(sr, fm, f"{statistics.median([kernel_time_us(run) for _ in range(3)]):.4f} us")
PY
```

Current readings: RTNE **56.95** / **42.55**, SR **77.33** / **57.77**.

Two-oracle suite — 41 triton-parity (36 RTNE + 5 linear SR) and 54 TE-reference:

```bash
pytest -q test/prototype/moe_training/nvfp4_training/{test_quantize_2d,test_hadamard_quantize_row_col,test_group_rht_quantize_row_col,test_group_quantize_2d}.py \
  -k "test_cutedsl_weight_quantize_2d_matches_triton or test_cutedsl_vs_triton_interchangeable or test_cutedsl_group_quantize_matches_triton_bitwise or test_cutedsl_group_quantize_2d_matches_triton or test_cutedsl_vs_triton_stochastic_rounding_bitwise"
pytest -q test/prototype/moe_training/nvfp4_training/{test_quantize_2d,test_hadamard_quantize_row_col,test_group_rht_quantize_row_col,test_group_quantize_2d}.py \
  -k "(vs_transformer_engine_reference or test_group_rht_correctness or test_group_rht_deepseek_dimensions_correctness) and cutedsl"
```

SR gate — 24 cases:

```bash
pytest -q test/prototype/moe_training/nvfp4_training/{test_group_rht_quantize_row_col,test_hadamard_quantize_row_col}.py \
  -k "test_cutedsl_vs_triton_stochastic_rounding_bitwise or test_cutedsl_rht_quantize_sr_unbiased or test_group_rht_sr_reconstructs or test_group_rht_sr_unbiased or (test_group_rht_rng_state_controls_stochastic_rounding and cutedsl) or (test_group_rht_stochastic_rounding_launches and cutedsl)"
```

SASS recipe: `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=<dir>` around the compile call, then
`cuobjdump -res-usage -sass <dir>/*.cubin`. Count by base family with
`grep -oP '^\s+/\*[0-9a-f]+\*/\s+(@!?U?P[T0-9]+\s+)?\K[A-Z][A-Z0-9._]*' | sed 's/\..*//' | sort | uniq -c`.
Do **not** count with `grep -c '^FMUL'` — it also matches `FMUL2`, and `^FMNMX` matches
`FMNMX.NAN`/`FMNMX3.NAN`.

## Surgical Simplicity

- `_mul_cvt_rs_e2m1x8_f32`, `_mul_cvt_rs_e2m1x8_acc_f32`, `_pack16_rs_from_enc`: three new
  functions that let the next commit delete six older ones; net 186 lines removed.
- `philox4_all`: a second generator is required because the linear path must keep the
  discarding one for triton parity; both have callers.
- `TILE_BLOCKS`: named constant beside `TILE_PACKED`, used at both epilogue call sites.
- `sqnr_floor` parameter: one default argument, added because SR genuinely cannot meet the
  RTNE bar and hardcoding a second threshold would have duplicated the reference helper.
- Two new grouped SR tests: they replace 4 deleted cases and cover two distinct failure
  modes (reconstruction error, stream bias) that nothing else guards.

## Confidence / Risk

- **HIGH** that the wins are real: measured on the canonical sentinel, corroborated by an
  independent static instruction count that moved in the predicted direction and
  magnitude, and reproduced in a full re-measurement against TE.
- **HIGH** that RTNE did not regress: 41 + 54 bitwise cases pass unchanged, including the
  TE fast-math byte-identity test.
- **MEDIUM-HIGH** on the new SR stream's statistical quality. Unbiasedness and SQNR are
  verified at K=32 draws on one shape; a correlation between the column and row streams at
  coinciding counters is argued from distinct keys rather than measured.
- **Risk LOW-MEDIUM.** The behaviour change is deliberate, documented in three places, and
  isolated to grouped SR. Anyone depending on grouped SR being bitwise-equal to triton
  will now fail — that is the intended contract change, and the docstrings say so.
