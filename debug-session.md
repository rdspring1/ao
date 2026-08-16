# Debug Session: TE E=4 Sequential Timing

## Experiment 1

- Hypothesis: Four validated TE single-expert calls can provide a same-work E=4 device-time baseline for grouped TorchAO kernels.
- Action: Ran the targeted TE-default timing script from `/home/me/nvfp4`.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` with the E=4 timing script shown in the session log.
- Result: Failed during imports with `ModuleNotFoundError: No module named 'benchmarks.prototype'`; no benchmark ran.
- Interpretation: The torchao `benchmarks` package is checkout-local and not editable-installed. Add `/home/me/nvfp4/third_party/torchao` to `sys.path`, matching the existing TE harness.
- Canonical-command status: The existing harness performs this path insertion explicitly.
- Failure classification: invocation mismatch.

## Experiment 2

- Hypothesis: Adding the torchao checkout to `sys.path` will make the same E=4 TE timing experiment runnable.
- Action: Re-ran the same script with `/home/me/nvfp4/third_party/torchao` prepended to `sys.path`.
- Result: Passed. Gate/up: RHT 59.210 us, 2D weight 60.489 us. Down: RHT 59.086 us, 2D weight 60.617 us.
- Interpretation: Four validated TE-default single-expert calls provide a same-work E=4 fallback baseline. It is not TE's grouped API and must remain labeled `TE 4xsingle`.
- Canonical-command status: Matches the path setup and timing helper used by the existing TE harness.
- Failure classification: resolved invocation mismatch.

## Experiment 3

- Top hypothesis: The grouped TE harness passes row-prefix `tensor_offsets`, but TE requires element-prefix offsets (`cumsum(first_dims * logical_last_dim)`). Source: TE `common.h` and `ep.py` use `tex.splits_to_offsets(first_dims, hidden)`.
- Strongest competing hypothesis: Grouped and single TE quantizers use different RHT sign masks.
- Untested assumption: Correct element offsets make E=1 grouped TE reproduce its single-tensor output byte-for-byte.
- Best next experiment: Compare E=1 grouped and single TE outputs with `tensor_offsets=tex.splits_to_offsets(first_dims, hidden)`.
- Missing source data: none.

### Result

With `tensor_offsets=tex.splits_to_offsets(first_dims, hidden)`, E=1 grouped TE matched
single-tensor TE exactly for row/column codes and scales. E=4 grouped TE then matched both
TorchAO backends with 0% differing bytes across the same four outputs. The competing sign
mask hypothesis is rejected.

## Experiment 4

- Hypothesis: Corrected grouped metadata enables a valid same-process E=4 timing comparison.
- Action: Timed grouped Triton, grouped CuteDSL, and grouped TE default on both 671B shapes.
- Result: Gate/up 92.820 / 62.367 / 47.930 us; down 92.297 / 64.122 / 48.003 us.
- Interpretation: Grouped TE is valid and is the RHT comparator. Four single TE calls remain necessary only for 2D weight quantization.
- Failure classification: resolved harness bug.

## Experiment 5

- Hypothesis: Replacing opaque inline PTX `div.rn.f32` with CuteDSL's compiler-visible FP32 division preserves TE-default RTNE bits while allowing better scheduling.
- Action: Ran one midpoint-sensitive linear TE-reference case and one grouped TE-reference case after changing only `_div_rn_f32`.
- Exact command: `pytest -q 'test/prototype/moe_training/nvfp4_training/test_hadamard_quantize_row_col.py::test_rht_quantize_rtne_vs_transformer_engine_reference[bounded_integer-256x256-cutedsl]' 'test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py::test_group_rht_correctness[seed223-cutedsl]'`
- Result: The linear case passed. The grouped case passed its row/column scale and code bitwise assertions, then failed in the subsequent independent cross-check with `NameError: name 'group_tensors' is not defined` at test line 370.
- Interpretation: The candidate preserves the targeted TE-default numerics. The failure is a pre-existing test-body defect after the relevant assertions, not a kernel result; do not broaden this optimization to fix it.
- Canonical-command status: Canonical targeted pytest cases; grouped case is partially unusable because of the unrelated test defect.
- Failure classification: pre-existing test defect after successful kernel validation.

### Performance result

- Exact command: `env -u NVTE_USE_FAST_MATH python -` importing `time_grouped_rht` and timing only E=4 `(2048, 7168)` and `(7168, 2048)`.
- Result: Candidate CuteDSL measured 64.857 / 66.796 us versus the prior 62.367 / 64.122 us; TE remained 48.076 / 48.007 us.
- Interpretation: Compiler-visible scalar division is slower, not faster. Reject the hypothesis and restore explicit `div.rn.f32`.
- Failure classification: supported optimization hypothesis rejected by performance validation.

## Experiment 6

- Hypothesis: Batching per-block exact reciprocals ahead of FP4 conversion will expose the same division ILP used by TE's vector scale phase.
- Action: Split grouped row/column epilogues into a scale pass and quantization pass, rereading each fragment in the second pass.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` running `compare_grouped(4, 2048, 7168)` and `time_grouped_rht` for only the two E=4 671B shapes.
- Result: All TE comparisons remained at 0% differing bytes. CuteDSL regressed to 82.717 / 84.739 us from 62.367 / 64.122 us.
- Interpretation: Duplicate TMEM/SMEM fragment reads overwhelm any division overlap. Retain the batching hypothesis for one discriminating variant that keeps fragments in the epilogue's allocated registers; stop this experiment family if register pressure also loses.
- Failure classification: implementation layout rejected; arithmetic hypothesis remains supported by serialized SASS.

### Register-retention result

- Exact command: `env -u NVTE_USE_FAST_MATH python -` running `time_grouped_rht(4, 2048, 7168)` only.
- Result: Retaining all eight column fragments and four row fragments increased CuteDSL to 94.870 us.
- Interpretation: Register pressure is worse than rereading. Reject division batching within the current warp-owned epilogue and restore the original code.
- Failure classification: optimization family rejected for the current epilogue architecture.

## Experiment 7: Direct TE Source Comparison

- Observed: TE column copies the complete TMEM partition to registers, fences, and releases the accumulator pipeline before BF16 rounding, amax, division, and conversion (`graph_safe_group_row_cast_col_hadamard_transform_cast_fusion.cu:819-925`).
- Observed: CuteDSL loads and quantizes eight 16-value TMEM fragments serially, releasing the accumulator pipeline only afterward (`_cutedsl_group_kernels_impl.py:754-783`).
- Observed: TE row copies the complete SMEM partition and releases the mainloop pipeline before its per-vector quantization loop (`.cu:1041-1069`); CuteDSL releases only after four load/quantize passes (`.py:872-910`).
- Interpretation: The failed register-retention experiment preserved CuteDSL's late release and duplicated a 16-value temporary, so it did not reproduce TE's defining lifetime/layout optimization. The next change should directly port compact bulk fragment ownership and early release.

## Experiment 8

- Hypothesis: Merging the eight UMMA-N modes and asking CuteDSL's generic TMEM selector for a 128x128 epilogue will reproduce TE's bulk column fragment.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` running only `time_grouped_rht(4, 2048, 7168)`.
- Result: Compile-time failure before GPU execution. The selected partition had shape `(((128, 32), 1), 1, 1)` (4096 values per thread), not the required 128, so reshaping to `(8, 16)` failed.
- Interpretation: TE uses the explicit `SM100_TMEM_LOAD_32dp32b64x` atom; the generic helper cannot infer that atom from the enlarged tile with the current layout. Select the equivalent explicit CuTeDSL atom rather than varying layout guesses.
- Canonical-command status: Canonical one-shape timing entry; compilation failed before timing.
- Failure classification: incorrect TMEM copy atom selection.

### Layout follow-up

- Selecting FP4 as the destination type did choose the TE-equivalent 32-data-path atom, but the partition remained 4096 values per thread.
- Compile-time layouts showed transformed TMEM `((128,1),((16,1),8),4)` with strides `((65536,0),((1,0),16),128)`. Merging N/U produced logical `(128,128,4)` but lost the data-path partition required by the atom; leaving it nested caused `flat_divide` to leave an extra eight-mode and was not congruent with the tile slice.
- This is the third occurrence of the same bulk-TMEM-layout blocker. The working scalar-fragment kernel was restored per the debug churn guard.

## Experiment 9: Direct MMA Bulk Fragment

- Hypothesis: `partition_shape_C((128, 128))` can alias the existing TMEM allocation with TE's full-fragment ownership and `SM100_TMEM_LOAD_32dp32b64x` equivalent.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` invoking one E=1 `(128, 128)` zero-input CuteDSL grouped quantization.
- Result: The layout probe succeeded: the bulk fragment is `((128,16),1,8,4):((65536,1),0,16,128)`, exactly matching the existing physical offsets with its zero-stride mode removed. The x64 copy partition exposes 64 values x 2 residual tiles per thread (128 total). The first production compile then failed only because CuTe tensor indexing rejects a Python `slice` on the loaded register tensor.
- Interpretation: The prior 4096-size reading incorrectly counted the copy atom's 32 collective data paths as per-thread values. The layout/copy hypothesis is supported; reshape the register tensor to `(8, 16)` and select a block with a CuTe coordinate.
- Canonical-command status: Compile-only minimal shape; no performance or correctness test ran.
- Failure classification: register-tensor indexing API mismatch.

### Bulk-load result

- Gate/up timing improved from 62.367 to 60.976 us (TE 47.830 us; Triton 92.805 us).
- Actual grouped TE comparison failed for CuteDSL column output: codes 99.4856% different and scales 82.4229% different; row codes/scales remained 0% different.
- Interpretation: The early-release structure is measurably faster, but the x64 register fragment's logical ordering is not eight contiguous 16-value blocks. Do not use this commit as a correct kernel; the next action is to derive the register permutation from the copy atom and restore TE-exact column ordering before any further timing.
- Failure classification: incorrect bulk-fragment register ordering.

## Experiment 10: CuTe Register-Mode Order

- Hypothesis: The x64 load already returns contiguous TMEM columns, but
  `reshape((8, 16))` treats the eight-block mode as CuTe's fastest mode and makes
  each selected block stride by eight registers.
- Evidence: `SM100_TMEM_LOAD_32dp32b64x` returns registers in increasing TMEM-column
  order, and CuTe compact layouts make mode 0 contiguous. The physical accumulator
  layout places each 16-token block contiguously at stride 16.
- Action: Reshape the 128-register fragment as `(16, 8)` and select `(None, u)` so
  each `_quant16` receives one contiguous token block.
- Success criterion: The focused grouped TE comparison reports 0% differing bytes
  for column codes and scales; do not benchmark before that passes.

### Focused-oracle result

- Exact command: `env -u NVTE_USE_FAST_MATH pytest -q 'test/prototype/moe_training/nvfp4_training/test_group_rht_quantize_row_col.py::test_group_rht_correctness[seed223-cutedsl]'`
- Result: All four TE-derived bitwise assertions passed. The test then reached the
  known unrelated `NameError: group_tensors` in its independent cross-check.
- Interpretation: The `(16, 8)` mode order restores the expected contiguous blocks.
  Run the actual grouped TE comparator once before accepting the fix.
- Failure classification: pre-existing test defect after successful kernel validation.

### Direct-TE invocation setup

- Exact command: `env -u NVTE_USE_FAST_MATH python -` importing
  `transformer_engine_torch` before the Python package.
- Result: Import failed with undefined symbol
  `_ZTIN18transformer_engine15CommOverlapBaseE`; no kernel ran.
- Interpretation: Initialize `transformer_engine.pytorch` before importing its extension,
  matching TE's supported import path.
- Failure classification: invocation mismatch.

### Direct grouped-TE result

- Exact command: `env -u NVTE_USE_FAST_MATH python -` comparing E=4, M=N=128
  CuteDSL outputs with `tex.split_quantize` grouped NVFP4 outputs.
- Result: Row codes differed in 0/32768 bytes and column codes in 0/32768 bytes.
  Naively flattened scale storage differed in 3665/4096 row bytes and 3656/4096
  column bytes.
- Interpretation: Exact codes confirm the register permutation. TE's opaque scale
  storage and TorchAO's logical blocked scale views require the established layout
  conversion before positional comparison; do not infer an arithmetic regression from
  naive flattening.
- Best next experiment: Compare scale bytes after applying the repo's blocked-layout
  transforms, with no kernel edit.

### Layout-aware grouped-TE result

- Exact command: `env -u NVTE_USE_FAST_MATH python -` on E=4, M=N=128, applying
  `to_blocked` to TE row scales and `to_blocked_grouped` to TE column scales.
- Result: Row codes, row scales, column codes, and column scales each differed in
  0 bytes.
- Interpretation: The register-mode fix is numerically exact against actual grouped
  TransformerEngine default output. The bulk-load optimization may now be benchmarked.
- Failure classification: supported hypothesis; implementation validated.

### Target performance result

- Exact command: `env -u NVTE_USE_FAST_MATH python -` calling `run_experiment` only
  for E=4 `(2048, 7168)` and `(7168, 2048)` RTNE.
- Result: CuteDSL measured 61.190 / 62.935 us; Triton measured 92.787 / 92.267 us.
- Interpretation: The corrected bulk load improves both CuteDSL targets by about 1.9%
  from the 62.367 / 64.122 us baseline.
- Failure classification: optimization validated.

## Oracle Move Validation

- Hypothesis: Moving the test-only oracle and updating its seven consumers preserves test
  collection and removes all imports of the old production module path.
- Exact command: `git diff --check && ! rg -n
  "torchao\\.prototype\\.moe_training\\.nvfp4_training\\.nvfp4_reference|from
  \\.hadamard_utils" test torchao benchmarks -g '*.py' && pytest --collect-only ...`
- Result: The guard matched legitimate relative `hadamard_utils` imports in three unrelated
  production modules, so pytest did not run.
- Interpretation: Narrow the relative-import check to the moved oracle file; retain the
  repository-wide check only for the removed fully qualified oracle path.
- Canonical-command status: pytest collection is canonical; the preceding custom guard was
  overly broad.
- Failure classification: invocation mismatch.

### Corrected oracle-move validation

- Exact command: `git diff --check`, narrow stale-path guards, and `pytest
  --collect-only -q` over the six consuming NVFP4 test modules.
- Result: No stale imports, clean diff check, and 570 tests collected successfully.
- Interpretation: The oracle move preserves all repository consumers and removes the
  test-only implementation from the production package.
- Failure classification: implementation validated.

## History Rewrite Validation: Direct TE Invocation

- Hypothesis: The rewritten Triton commit remains byte-identical to actual
  TransformerEngine default for single RHT and 2D weight outputs.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` invoking TE `NVFP4Quantizer`
  and the rewritten Triton kernels for a 256x256 BF16 tensor.
- Result: Failed before the TorchAO kernel launched because
  `triton_rht_quantize_row_col` received the sign vector in the
  `col_global_amax` position.
- Interpretation: Invocation mismatch; the custom-op schema requires
  `(A, col_global_amax, row_global_amax, sign_vector, stochastic_rounding)`.
- Canonical-command status: Direct comparator is appropriate; correct only the argument
  order before rerunning.
- Failure classification: invocation mismatch.

## History Rewrite Validation: Ruff

- Hypothesis: The rewritten first-layer files satisfy the repository's pinned Ruff rules.
- Exact command: `ruff check` followed by `ruff format --check` on the eleven files in
  the Triton/oracle commit.
- Result: `ruff check` reported five `I001` import-order findings; formatting did not run
  because the check exited nonzero.
- Interpretation: The findings are a single mechanical import-order family and are all
  reported fixable by Ruff. Applied the documented targeted `ruff check --fix`, formatted,
  and revalidated successfully with Ruff 0.11.6.
- Canonical-command status: Uses Ruff 0.11.6 pinned by `CONTRIBUTING.md`.
- Failure classification: resolved implementation formatting defect.

## Rewritten CuTeDSL Direct-TE Validation

- Hypothesis: Both rewritten backends remain byte-identical to actual TE default for
  single RHT, grouped RHT E=4, and 2D weight quantization.
- Exact command: `env -u NVTE_USE_FAST_MATH python -` comparing both backends with TE
  outputs at 256x256 and grouped E=4 128x128.
- Initial result: Failed in comparator setup before any TorchAO kernel call because a
  hard-coded `(2, 2, 32, 16)` view contained 2048 elements while TE exposed 4096 scale
  bytes.
- Interpretation: Invocation-harness shape error. Derived each blocked-scale view from
  the actual TorchAO output shape and reran without changing kernels.
- Corrected result: Triton and CuTeDSL each reported zero differing bytes for row codes,
  row scales, column codes, and column scales in all three operations.
- Failure classification: resolved invocation mismatch; implementation validated.
