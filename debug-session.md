# CuTeDSL Quantization Debug Session

## Experiment 1 — 1D linear 128-row supertile

- Hypothesis: the existing 128-row RHT geometry (320 threads/CTA) would improve
  occupancy over the profiled 256-row geometry (448 threads/CTA) on the
  `(2048, 7168)` gate/up sentinel.
- Action: selected `col_groups_per_supertile = 8` only when `apply_rht=True`.
- Command: `python - <<'PY' ... kernel_time_us(cutedsl_rht_quantize_row_col)` with
  seed 123, BF16 `(2048, 7168)`, default math, RTNE; three profiler samples, each
  using the canonical 15 warmups and 50 measured iterations.
- Result: baseline `23.6610 us`; experiment `23.6514`, `23.6391`, `23.6493 us`.
- Interpretation: median gain was about 0.05%, far below the approximately 2%
  acceptance threshold. The 448-thread launch is not the limiting factor for this
  sentinel.
- Canonical-command status: canonical sentinel shape and timing helper; direct
  one-off driver rather than the multi-shape benchmark CLI.
- Failure classification: low-information performance result; change rejected and
  reverted with an inverse patch.

## Experiment 7 — Shared packed BF16-to-FP4 epilogue

- Hypothesis: the packed multiply/convert primitive that was sub-threshold when only
  the 1D row half used it would exceed the threshold when both halves of the 2D
  no-MMA kernel used it.
- Evidence: TransformerEngine's exact path uses packed `mul.f32x2` immediately before
  packed FP4 conversion; both 2D row and column epilogues consume BF16-origin values.
- Action: added `_mul_cvt_rn_e2m1x8_f32` and selected it for exact-math RTNE blocks
  whose inputs are already BF16 values widened to FP32.
- Command: canonical direct RTNE `cutedsl_weight_quantize_2d` sentinel at
  `(2048, 7168)`, three samples with 15 warmups and 50 timed profiler iterations.
- Result: `23.4702`, `23.4453`, and `23.4433 us` versus `24.7719 us`, about 5.3%
  faster. Final matrix median was `23.2973 us`, 5.95% faster than the saved baseline.
- Interpretation: eliminating the intermediate scalar scaled-value tensor reduces
  epilogue work enough when both output orientations benefit.
- Correctness: smallest 2D linear, 1D linear, and 1D grouped bitwise cases passed;
  grouped 2D bitwise parity also passed.
- Canonical-command status: canonical shape and timing parameters; direct driver
  rather than multi-shape CLI.
- Failure classification: none; validated optimization retained and committed as
  `4a6ec376`.

## Experiment 2 — 1D linear packed multiply/convert

- Hypothesis: matching TransformerEngine's packed `mul.f32x2` plus FP4 conversion
  sequence in the plain BF16 row epilogue would reduce scalar arithmetic and register
  pressure enough to improve the `(2048, 7168)` sentinel.
- Evidence: installed TransformerEngine
  `common/util/ptx.cuh::mul_cvt_bf16_to_fp4_8x_round_to_nearest` uses four packed
  FP32x2 multiplies followed immediately by four packed FP4 conversions.
- Action: added one inline CuTeDSL primitive for eight values and used it only for
  exact-math RTNE non-RHT blocks.
- Command: the same canonical direct sentinel driver as Experiment 1; one initial
  sample followed by two confirmation samples.
- Result: `23.2968`, `23.3767`, and `23.3460 us` versus baseline `23.6610 us`, a
  roughly 1.2-1.5% gain.
- Interpretation: the hypothesis was directionally supported but did not meet the
  approximately 2% acceptance threshold. Retaining a shared primitive would also
  broaden 2D scope without sufficient payoff.
- Canonical-command status: canonical shape, math, rounding, warmups, and timed
  iterations; direct one-off driver rather than the multi-shape benchmark CLI.
- Failure classification: sub-threshold performance improvement; change rejected and
  reverted with an inverse patch.

## Experiment 3 — 1D linear column-store overlap

- Hypothesis: double-buffering the column FP4/scale staging and changing the TMA wait
  from `wait_group(0)` to `wait_group(1)` would overlap the prior iteration's output
  store with the next persistent iteration.
- Evidence: the row epilogue already uses this safe two-stage pattern, while the
  column epilogue used one buffer and fully drained each store.
- Action: added two stages to both column output buffers, selected them by persistent
  iteration, limited in-flight stores to one, and drained the final store at exit.
- Command: canonical direct default-math RTNE `(2048, 7168)` sentinel with 15 warmups
  and 50 measured profiler iterations.
- Result: `23.7940 us` versus baseline `23.6610 us`, about 0.6% slower.
- Interpretation: with only three persistent iterations at this shape, extra shared
  memory and staging overhead outweighed the store overlap.
- Canonical-command status: canonical sentinel shape and timing parameters; direct
  driver rather than multi-shape CLI.
- Failure classification: performance regression; change rejected and reverted with
  an inverse patch.

## Experiment 5 — 1D grouped 16-tile scheduler work items

- Hypothesis: increasing scheduler work items from 8 to 16 token tiles would halve
  CLC fetches and group searches for 2048-token experts.
- Evidence: each expert has exactly 16 128-token tiles, so the sentinel currently uses
  two work items per hidden tile and never crosses an expert boundary within either.
- Action: changed `K_TILE_MAX` from 8 to 16.
- Command: the same canonical grouped sentinel as Experiment 4.
- Result: `77.6977 us` versus baseline `61.3872 us`, about 26.6% slower.
- Interpretation: the longer work item substantially reduces persistent load balance;
  the TransformerEngine-derived 8-tile setting is necessary despite scheduler costs.
- Canonical-command status: canonical sentinel shape and timing parameters; direct
  driver rather than multi-shape CLI.
- Failure classification: major performance regression; change rejected and reverted
  with an inverse patch.

## Experiment 6 — 2D linear 128-row supertile

- Hypothesis: the existing 128-row no-MMA geometry (416 threads) would schedule more
  efficiently than the 256-row geometry (544 threads) on `(2048, 7168)`.
- Evidence: the no-MMA path assigns 256 column threads plus one row thread per
  supertile row, making its 256-row block much larger than the profiled RHT block.
- Action: selected the 128-row geometry for `apply_rht=False`.
- Command: canonical direct RTNE `cutedsl_weight_quantize_2d` sentinel at
  `(2048, 7168)`, 15 warmups, and 50 measured profiler iterations.
- Result: `24.7887 us` versus baseline `24.7719 us`, effectively flat/slightly slower.
- Interpretation: halving tile height does not improve the no-MMA path; the extra tile
  count offsets its smaller block.
- Canonical-command status: canonical sentinel shape and timing parameters; direct
  driver rather than multi-shape CLI.
- Failure classification: low-information performance result; change rejected and
  reverted with an inverse patch.

## Experiment 4 — 1D grouped aligned row stores

- Hypothesis: replacing the two aligned 32-bit row-code stores with one CuTeDSL
  autovectorized two-word copy would reduce global-store instructions.
- Evidence: each thread owns two contiguous u32 words at an 8-byte-aligned block
  offset, while the column path already uses `cute.autovec_copy` for contiguous words.
- Action: staged the two words in a two-element register tensor and autovector-copied
  them into an aligned two-word local tile, including the capacity-zero path.
- Command: canonical direct default-math RTNE grouped sentinel with `E=4`, each
  `(2048, 7168)`, 15 warmups, and 50 measured profiler iterations.
- Result: `62.2391 us` versus baseline `61.3872 us`, about 1.4% slower.
- Interpretation: the register tensor/copy lowering costs more than the two direct
  stores; alignment was not the limiting issue.
- Canonical-command status: canonical sentinel shape and timing parameters; direct
  driver rather than multi-shape CLI.
- Failure classification: performance regression; change rejected and reverted with
  an inverse patch.

## Experiment 8 — SASS-first triage of the 2D weight kernel

- Hypothesis: after the packed-convert win, the remaining TE gap is either epilogue
  instruction count or occupancy, and the compiled SASS can tell them apart without
  spending a benchmark.
- Action: `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=... ` around `_compile_fused_kernel(
  0, True, False, apply_rht=False, grouped=False, col_groups_per_supertile=16)`, then
  `cuobjdump -res-usage -sass`.
- Result: `REG:49 STACK:0`, 3432 static instructions, of which 256 `LDS.U16`, zero
  vector loads, and roughly 1823 (53%) integer/address arithmetic.
- Interpretation: occupancy is not register-limited, so the 544-thread block is not
  the binding constraint. The kernel is dominated by per-element swizzled address
  computation feeding scalar 16-bit SMEM loads. This redirected the whole round away
  from geometry and toward the SMEM read shape.
- Failure classification: none; this is the discriminating measurement the round was
  built on.

## Experiment 9 — Vectorized rowwise SMEM reads

- Hypothesis: the row epilogue's 16 scalar indexed reads per block are contiguous
  (the `MN_SW128` atom makes the N grain stride-1) and should be one vector copy.
- Evidence: `_cutedsl_group_kernels_impl.py:891-895` already reads its row blocks with
  `cute.autovec_copy` over the same atom and the same `tile_to_shape` clean layout;
  only the linear kernel still used scalar reads. TE's 2D kernel likewise issues
  `LDS.128` for its rowwise half and scalar `LDS` only for the transposed half.
- Action: replaced the `for j in range(16)` scalar read at `_cutedsl_kernels_impl.py`
  with `cute.autovec_copy(cute.local_tile(sA_clean[(None, k_row, stage)], (16,), (b,)),
  rBlk)` and hoisted `blk`/`rBlk` out of the block loop.
- Result: 128 `LDS.U16` -> 16 `LDS.128`; 3432 -> 3048 static instructions; REG 49 ->
  51. 2D linear `(2048, 7168)` `23.2973 -> 18.3297 us` (21.3%), 2D grouped E=4
  `87.3549 -> 67.8821 us` (22.3%), 1D linear RTNE `23.2411 -> 21.4503 us` (7.7%).
- Correctness: 45 Triton-parity cases and 54 TE-reference CuteDSL cases all passed.
- Failure classification: none; retained.

## Experiment 10 — Hoisting the col N-row slice

- Hypothesis: the column epilogue's per-element swizzled addressing would strength
  reduce if the N-row and stage were sliced out once per iteration, mirroring what
  made the row half cheap.
- Action: `sA_col = sA_clean[((nrow % 64, nrow // 64), None, (0, stage))]` hoisted out
  of the 16-element loop, indexed by `mpos` inside.
- Result: 3072 static instructions versus 3048; slightly worse.
- Interpretation: the DSL and ptxas already strength-reduce this addressing. The col
  half's cost is the scalar 16-bit access width itself, not redundant address math.
- Failure classification: low-information result; reverted before measuring.

## Experiment 11 — Paired columnwise SMEM reads

- Hypothesis: a column thread that owns an adjacent N-row pair reads 32 bits instead
  of 16, so a warp moves the full 128 B instead of 64 B, and the pair shares its
  16x16 block amax, scale and reciprocal.
- Evidence: TE's tuned 1D kernel does exactly this
  (`specialized/quantize_transpose_nvfp4_tuned_1D.cuh:188-268`): column pairs read
  with `ld_shared_b32`, two block amaxes taken from one `bf16x2` accumulator.
- Action: `nrow = (tidx % 64) * 2`, `u_quarter = tidx // 64`, 4 col-groups per thread,
  `cute.autovec_copy` of a 2-element tile per M-position; `_group16_amax` parameterized
  to take the 8-lane offsets `(4, 2, 1)` after an in-register fold; `_quant16_from_amax`
  split into `_enc_from_amax` + `_pack16_rn_from_enc` so the E4M3 round-trip and the
  exact reciprocal are computed once for the pair.
- Result: 128 `LDS.U16` -> 64 32-bit `LDS`; 3048 -> 2712 static instructions; REG 51 ->
  56. 2D linear `18.3297 -> 16.4788 us` (10.1%), 2D grouped `67.8821 -> 60.8464 us`
  (10.4%).
- Correctness: 61 combined 2D Triton-parity and TE-reference cases passed.
- Failure classification: none; retained.

## Experiment 12 — Fused RHT-accumulator epilogue

- Hypothesis: the `rht_acc` exact path is the last place the packed multiply/convert
  win was not applied, and folding its bfloat16 round-through into the same asm block
  should give the 1D column epilogue what the 2D epilogues already have.
- Evidence: TE's `ptx.cuh:824` consumes bfloat16 directly and relies on
  `cvt.rn.satfinite` for saturation rather than an explicit clamp.
- Action: added `_mul_cvt_rn_e2m1x8_acc_f32` (four `cvt.rn.bf16x2.f32`, shift/mask
  re-widen, four `mul.f32x2`, four `cvt.rn.satfinite.e2m1x2.f32`), selected via
  `_pack16_rn_from_enc(..., rht_acc)`. Dropped the `min.xorsign.abs` clamp.
- Result: 1D linear RTNE `21.4503 -> 19.5726 us` (8.8%); 1D grouped inherits it via the
  shared `_quant16`, `61.6432 -> 57.5215 us` (6.7%).
- Correctness: 33 cases including the TE fast-math byte-identity test and the
  zero/near-zero no-saturation test (which is what exercises the removed clamp).
- Failure classification: none; retained.

## Experiment 13 — Two co-resident CTAs on the 2D path

- Hypothesis: with 56 registers and a ~92 KiB SMEM budget at the 128-row supertile,
  two 416-thread CTAs fit per SM, raising 17 warps/SM to 26.
- Evidence: all TMEM allocation is `apply_rht`-gated, so the weight path holds no
  TMEM and two CTAs cannot contend for it. The earlier 128-row experiment kept
  `GRID = min(NUM_SMS, num_super)` and therefore never tested occupancy.
- Action: forced `col_groups_per_supertile = 8` for `apply_rht=False` and doubled the
  persistent grid.
- Result: `16.1811 us` versus `16.4788 us`, 1.81% faster.
- Interpretation: real but below the ~2% bar, and it would pin every weight shape to
  the 128-row geometry. Occupancy is a genuine second-order effect here, not the
  first-order one the pre-round table suggested.
- Failure classification: sub-threshold improvement; reverted.

## Experiment 14 — SR vs RTNE SASS diff (no benchmark)

- Hypothesis: the grouped SR gap has three candidate causes -- discarded Philox words,
  an unfused multiply/clamp, and register spilling against the shared `REG_COL` /
  `REG_ROW` budget -- and a static instruction diff can size all three for free.
- Evidence: `philox4:397` runs four full `philox_c0` schedules per 16-element block and
  keeps only `c0` of each; `_quant16_from_amax:776` gates the fused packing on
  `not sr and not fast_math`; `REG_DEALLOC/REG_COL/REG_ROW` are module-level ints shared
  by every compiled variant.
- Action: compiled `_compile_group_fused_kernel(0, True, sr, False)` for both `sr`
  values under `CUTE_DSL_KEEP=cubin CUTE_DSL_DUMP_DIR=`, then `cuobjdump -res-usage
  -sass` and diffed the opcode mix by base family.
- Result: SR 5816 static instructions vs RTNE 3456, +2360. `IMAD` +1044 and `LOP3` +758
  (1802, 76%); `FMUL` +192 and `FMNMX` +192 (384, 16%). `IMAD.HI.U32` goes 0 -> 536.
  Both report `REG:128`, `LDL` 14 -> 16, `STL` 8 -> 9, no `MOV.SPILL`.
- Interpretation: the RNG is the dominant cost and the unfused arithmetic is second.
  The spill hypothesis is refuted -- SR does not spill where RTNE does not -- so the
  planned SR-specific register-budget round was dropped before it was written.
- Failure classification: none; diagnostic only, no source change.

## Experiment 15 — Fused SR multiply and clamp

- Hypothesis: giving SR the same fused multiply/convert RTNE already has removes the
  384 `FMUL`/`FMNMX` Experiment 14 measured.
- Evidence: TE fuses the multiply into the same asm block and applies no clamp,
  relying on `.satfinite` (`ptx.cuh:940-990`); Experiment 12 already validated exactly
  that removal for RTNE.
- Action: added `_mul_cvt_rs_e2m1x8_f32` and `_mul_cvt_rs_e2m1x8_acc_f32` (two
  `cvt.rs.satfinite.e2m1x4.f32` in place of four `cvt.rn...e2m1x2`, one random word
  each) plus `_pack16_rs_from_enc`, and routed the `sr` branch through it.
- Discriminator for the dropped clamp: `cvt.rs` perturbs the mantissa before saturating,
  so `|x| > 6` could have diverged. The linear path retains full triton SR bitwise
  parity and is a real oracle for it -- all five
  `test_cutedsl_vs_triton_stochastic_rounding_bitwise` cases passed unchanged, so no
  explicit `min.xorsign.abs` was needed.
- Result: SR standard `150.73 -> 139.43 us` (7.5%), SR fast `116.45 -> 108.98` (6.4%);
  RTNE unchanged. `FMUL`/`FMNMX` excess over RTNE went +192/+192 -> 0/0.
- Correctness: bitwise neutral; 45 triton-parity + 54 TE-reference cases passed.
- Failure classification: none; retained.

## Experiment 16 — Fused RTNE fast-math path

- Hypothesis: the same gate left fast-math RTNE materializing the scalar tensor, and
  fixing it is bitwise-neutral.
- Action: collapsed `_quant16_from_amax` to one rounding-mode branch over a shared
  shape, with `use_acc = rht_acc and not fast_math` -- fast math takes the plain
  primitive even for an RHT accumulator, because it deliberately skips the bfloat16
  round-through. Deleted the six helpers this made unreachable (186 lines).
- Result: RTNE fast `45.60 -> 42.55 us` (6.7%); all other cases unchanged.
- Correctness: 45 + 54 cases, including
  `test_cutedsl_group_fast_math_matches_transformer_engine`, the byte-identity check
  against real TE fast math and the only direct guard on the dropped clamp here.
- Failure classification: none; retained.

## Experiment 17 — One Philox draw per block (grouped)

- Hypothesis: consuming all four Philox words per draw instead of one cuts the RNG cost
  Experiment 14 measured by roughly 3.6x.
- Evidence: TE issues one `generate4` per block for ~40 multiplies
  (`curanddx.hpp:36-101`, `ptx.cuh:914-990`); `philox4`'s 4-counter stride exists only
  to reproduce triton's `tl.randint`-per-packed-byte stream, which the user approved
  dropping for the grouped kernel.
- Action: added `philox4_all`, which reuses the existing `philox_prep` hoist and differs
  from `philox_c0` only in computing all four words in the last round: 34 multiplies for
  four words against 124. Counter derived from tile coordinates -- the previous
  packed-byte expression divided by 8, every term dividing exactly -- because the CLC
  scheduler makes visit order unstable and a running counter would make output depend on
  scheduling. Added `TILE_BLOCKS`; column and row keep distinct keys.
- Result: SR standard `138.34 -> 77.33 us` (44.1%), SR fast `108.75 -> 57.77` (46.9%).
  Over the three commits: `150.73 -> 77.33` (1.95x) and `116.45 -> 57.77` (2.02x).
  Static excess over RTNE `+2360 -> +736`; `IMAD` `+1044 -> +346`, `LOP3` `+758 -> +206`.
- Correctness: first non-bitwise-preserving change in this project. The four grouped SR
  bitwise cases asserted exactly the dropped property and were replaced by
  `test_group_rht_sr_reconstructs` and `test_group_rht_sr_unbiased`;
  `test_group_rht_rng_state_controls_stochastic_rounding` is unchanged and is the real
  guard on the new counter derivation. Full nvfp4_training suite: 594 passed, 36 skipped.
- Gap found while writing the tests: SR measures 17.2 dB reconstruction SQNR against
  RTNE's 20+, on both backends and all four fixtures within 0.2 dB. The unmodified
  triton backend measures the same, so the planned "SQNR at the RTNE threshold" was the
  wrong contract for a stochastic kernel rather than a defect -- SR's error is uniform
  over the quantization interval instead of bounded by half of it, about 3 dB. The bar
  was parameterized and set to 15 dB for SR.
- Failure classification: none; retained.

## Experiment 18 — Post-round CuteDSL vs TransformerEngine, E=4

- Action: re-measured both sides of the DSV3 671B FFN comparison at `c659013c`, TE
  2.19.0.dev0+172bd93, adding SR rows the table never carried.
- Result (complete pipeline, standard math): gate/up SR `TE 2.06x -> 1.21x` ahead, down
  SR `TE 2.12x -> 1.26x`. Fast math SR: `TE 1.11x` and `1.14x`.
- Interpretation: SR now costs CuteDSL 1.23x its own RTNE pipeline against TE's 1.37x,
  so the SR path is the more efficient of the two relative to its own baseline. The
  remaining gap is no longer SR-specific -- it is the RTNE gap, and it sits entirely in
  the quantize kernel, since the amax stage is at parity everywhere.
- Failure classification: none; measurement only.

## Experiment 19 — One Philox draw per block, linear path

- Hypothesis: the linear kernels already inherited the fused packing through the shared
  `_quant16_from_amax`, so the only grouped win they are missing is the RNG, and the same
  one-draw-per-block change should pay off at least as well.
- Evidence: linear SASS, SR vs RTNE. `FMUL` and `FMNMX` excess is already **0**, i.e.
  Experiments 15/16 did reach this path (linear SR `46.4310 -> 41.5547 us`, RTNE fast math
  `14.8403 -> 12.9741`, with no linear-specific change). `IMAD` +2007 and `LOP3` +1530 are
  3537 of the 3872-instruction excess, **91%** -- proportionally a larger Philox burden
  than grouped carried before Experiment 17 (76%).
- Blocker, and the decision: applying `philox4_all` here breaks the linear path's triton
  SR bitwise parity, which was the **only** bitwise oracle for stochastic rounding in the
  project -- there is no TE SR reference, since TE's stream differs by construction. It is
  also the oracle Experiment 15 used to settle the dropped clamp. Raised with the user
  with three options (apply and streamline / apply but keep a test-only parity variant /
  leave linear alone); the user chose to apply and streamline.
- Action: both linear epilogues switched to `philox4_all` with a coordinate-derived
  counter (`tile_id * TILE_BLOCKS + ...`, the old packed-byte expression divided by 8).
  Deleted `philox4`, `triton_tile_id`, `_GROUP_SIZE_N`, `TRITON_TILE_PACKED`; moved
  `TILE_BLOCKS` into the shared module, replacing the grouped module's copy and its now
  unused `TILE_PACKED`.
- Test replacement, which came out stronger than planned: rather than add a reconstruction
  test, `test_triton_rht_quantize_rs_at_most_one_fp4_step_from_rtne` already asserted that
  every SR code sits within one FP4 magnitude step of the RTNE code with matching signs.
  It ran on triton only; it now runs on **both backends** over one tile, several tiles and
  a short trailing column group. It pins SR against an RTNE code that is still bitwise
  checked against triton and TE, so nibble order, scale and block indexing are all still
  caught without any SR oracle. Net: one test and one fixture deleted, zero new tests.
- Result: linear SR standard `41.5547 -> 25.8337 us` (37.8%; `46.4310 -> 25.8337`, 44.4%
  across the whole series), SR fast `31.4567 -> 18.0922` (42.5%; 45.2% across the series).
  RTNE unchanged. Static excess over RTNE `+3872 -> +1112`, `IMAD` `+2007 -> +650`,
  `LOP3` `+1530 -> +423` -- a 71% cut, matching grouped's 69%.
- Correctness: 36 triton-parity + 54 TE-reference + 7 linear SR + 26 grouped SR.
- Process note: the first instinct was to re-run the whole `nvfp4_training` directory
  (~10 min, JIT-bound). The user pushed back twice. The genuine gap after the change was
  grouped SR alone, 18 s -- everything else was already covered by runs earlier in the
  session. Derive the blast radius from the diff before running.
- Failure classification: none; retained.
