# CuteDSL TE-Default RTNE Quantization Optimization

## Local NVFP4 History Rewrite Preflight

Status: IN PROGRESS

### Preflight

Preserve the current `bc87303f` lineage and workflow ledgers on
`nvfp4_moe_cutedsl_history`, then rebuild the local work as exactly one Triton/oracle
commit on `origin/nvfp4_moe` and exactly one CuTeDSL product commit above it. Repoint
only local branches; do not push, rewrite a remote, or use GitHub tooling.

### Observable Success Criterion

- `origin/nvfp4_moe..nvfp4_moe` contains one commit named
  `Match Triton NVFP4 kernels to TransformerEngine defaults`.
- `nvfp4_moe..nvfp4_moe_cutedsl` contains one commit named
  `Add CuTeDSL NVFP4 kernels`.
- The rewritten CuTeDSL tree equals the archived final product tree except for
  `save.md` and `debug-session.md`.
- Focused TE-default byte-equivalence and backend-equivalence validations pass, and the
  final worktree is clean.

### Scope Fence

The first commit contains only Triton numerical fixes, the test-only TE oracle and shared
assertions, group-aware blocked-scale support, and Triton-only oracle tests. The second
contains all remaining CuTeDSL integration and product changes. Preserve upstream branch
configuration and remove only the temporary rewrite branch after establishing final tips.

### Surgical Simplicity

No new source, API, abstraction, or test is being invented during the rewrite. The two
replacement commits partition the already validated final tree; the history branch is the
recoverable checkpoint for the original lineage and ledgers.

Status: COMPLETE — TEST-ONLY NVFP4 ORACLE MOVED UNDER TEST

### Preflight

Move the TE-derived PyTorch oracle from the production prototype package to the
equivalent NVFP4 test directory, update only its consumers and its runtime-helper import,
then validate test collection/imports. Preserve the oracle implementation unchanged.

### What Changed

Moved `nvfp4_reference.py` from the production prototype package to the equivalent test
directory. Updated all seven test consumers and changed the moved module's relative
`hadamard_utils` import to its production-qualified path. Oracle arithmetic is unchanged.

### Move Validation

- No imports of the removed production oracle path remain.
- `git diff --check` passed.
- All 570 tests in the six consuming test modules collected successfully.
- One initial stale-import guard was too broad and matched unrelated valid relative imports;
  `debug-session.md` records the corrected invocation.

### Surgical Simplicity

The file move is justified because all seven consumers are tests and no production or
benchmark module imports the oracle. No oracle logic, API, fixture, or test was added.

## Goal

Reduce the CuteDSL grouped RTNE RHT quantization gap against TransformerEngine default
at DeepSeek-V3 671B E=4 sizes without changing any TE-default output bits.

## Success Criterion

- Row/column FP4 codes and scales remain byte-for-byte equal to TE default.
- Gate/up and down E=4 CuteDSL latency materially improves from 62.367 / 64.122 us.
- Only the two requested performance shapes and focused correctness oracles run.

## Scope Fence

- Grouped CuteDSL RTNE RHT quantization only.
- Preserve exact `div.rn.f32` and BF16 RHT ordering.
- No stochastic-rounding work, Triton changes, full test sweep, or unrelated test repair.

## Preflight

Start from the committed TE-default implementation. Make one bounded scheduling change,
validate one midpoint-sensitive case and one grouped TE comparison, then benchmark only
the two E=4 shapes. Restore immediately on regression.

## Surgical Simplicity

The surviving kernel fix is a two-line register-layout correction on the already committed
bulk-load implementation. No new API, file, fixture, abstraction, or parameter was added.

## What Changed

- Corrected the source-derived bulk `SM100_TMEM_LOAD_32dp32b64x`-equivalent column load's
  register view from `(8, 16)` / `(u, None)` to CuTe's mode-0-contiguous `(16, 8)` /
  `(None, u)`. Each `_quant16` now receives its original contiguous 16-token block while
  retaining the early accumulator-pipeline release.

- Tested compiler-visible scalar division: bitwise correct, but 64.857 / 66.796 us.
- SASS showed each block's exact reciprocal serialized with its full conversion sequence.
- Tested eight/four-block scale batching with fragment rereads: bitwise correct, but
  82.717 / 84.739 us.
- Tested retaining all fragments in registers: gate/up regressed to 94.870 us.
- Exhaustively rejected replacing division with `global_encode * reciprocal(E4M3)`:
  1,746,152 of 4,112,514 pairs differed (42.459%).
- Restored both kernel files exactly to committed state after each regression.

## Validation

- Midpoint-sensitive linear TE-derived oracle passed for the scalar candidate.
- Grouped row/column code and scale bitwise assertions passed before a pre-existing
  `NameError: group_tensors` later in the test body.
- Actual grouped TE comparison for the batching candidate reported 0% differing bytes
  for all row/column codes and scales.
- The corrected bulk-load kernel reported 0 differing bytes for row codes, row scales,
  column codes, and column scales against actual grouped TransformerEngine default output.
- The focused grouped pytest passed all four TE-derived bitwise assertions before its
  pre-existing post-assertion `NameError: group_tensors`.
- Target-only performance validation measured 61.190 / 62.935 us, improving the committed
  62.367 / 64.122 us baseline by about 1.9% on both shapes.

## 671B TE-Default Performance

GB200 device kernel time, microseconds, `NVTE_USE_FAST_MATH` unset:

| Shape | Op | Triton | CuteDSL | TE |
|---|---|---:|---:|---:|
| gate/up, 2048x7168 | linear RHT quantize | 33.997 | 23.588 | 12.222 |
| gate/up, 2048x7168 | linear 2D weight | 45.627 | 24.599 | 13.597 |
| down, 7168x2048 | linear RHT quantize | 33.934 | 23.729 | 12.535 |
| down, 7168x2048 | linear 2D weight | 45.607 | 24.275 | 13.397 |
| gate/up, E=4 | grouped RHT amax | 40.018 | 22.862 | n/a |
| gate/up, E=4 | grouped RHT quantize | 92.820 | 62.367 | 47.930 |
| gate/up, E=4 | grouped 2D weight | 109.452 | 92.629 | 61.390* |
| down, E=4 | grouped RHT amax | 40.003 | 23.182 | n/a |
| down, E=4 | grouped RHT quantize | 92.297 | 64.122 | 48.003 |
| down, E=4 | grouped 2D weight | 108.508 | 91.717 | 60.705* |

Grouped RHT uses TE's validated grouped kernel. `*` TE 2D weight is the summed CUDA
self-time of four validated TE-default single-expert calls covering the same E=4 work;
TE 2.19 lacks grouped 2D weight quantization. TE amax-only remains unavailable.

## Next Action

Review and commit the test-only oracle relocation and updated imports.

## Debug Checkpoint

- Supported conclusion: the x64 atom returns increasing TMEM columns, and CuTe mode 0 is
  contiguous; `(16, 8)` is the required per-thread register view.
- Missing source data: none.
- Confidence: HIGH.
- Best next mode: vet or commit.

## Risk

LOW: the grouped pytest case still has an unrelated pre-existing post-assertion
`NameError`, but both its four relevant assertions and the independent actual grouped TE
comparison passed bitwise. Performance improvement is modest and may vary slightly by run.
