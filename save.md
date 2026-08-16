# TE-Default NVFP4 Division Alignment

Status: COMPLETE

## Goal

Make Triton and CuteDSL NVFP4 quantization follow TransformerEngine's default numeric recipe by using correctly rounded FP32 division throughout the scale chain, validate exact RTNE output against the TE-derived PyTorch oracle and actual TE, and measure DeepSeek-V3 671B performance at E=4 without running the full test matrix.

## Success Criterion

- Triton and CuteDSL use `div.rn.f32` for the final per-block encode reciprocal.
- CuteDSL also makes the existing global scale divisions explicitly `div.rn.f32`.
- A targeted midpoint-sensitive test establishes that the PyTorch oracle can assert RTNE FP4 codes bitwise.
- TE-default timings are collected for the 671B gate/up and down shapes; grouped E=4 timings are reported separately where TE has no validated equivalent.

## Scope Fence

- NVFP4 Triton and CuteDSL scale-chain arithmetic
- Existing TE-derived assertions and focused tests needed to express exact RTNE equality
- Targeted correction of the out-of-tree TE benchmark harness's grouped offset metadata
- Read-only execution of targeted correctness and benchmark commands
- No stochastic-rounding redesign, kernel scheduling changes, or full-suite execution

## Preflight

Use TE default numerics as the sole correctness target. Make one bounded arithmetic change, run one focused validation, enter hypothesis-first debug if it fails, and only benchmark after correctness passes. Do not run the full expensive matrix.

## Surgical Simplicity

One new internal numeric primitive is justified because both CuteDSL kernel modules need an explicit, instruction-level `div.rn.f32` contract. Existing tests and helpers will be revised in place; no new test file or public API is planned.

## What Changed

- Replaced Triton's final per-block `rcp.approx.f32` with `tl.div_rn` in the shared RHT helper and 2D weight kernel.
- Replaced CuteDSL's approximate reciprocal helper with explicit `div.rn.f32` and used it for all three TE scale-chain divisions in linear and grouped kernels.
- Tightened the TE-derived PyTorch oracle contract from bracketed FP4 codes to byte-for-byte RTNE code equality; removed the obsolete bracket machinery.

## Validation

- Static compilation and `git diff --check` passed.
- Actual TE-default conformance at 256x256 passed all 16 byte comparisons: Triton/CuteDSL, RHT/2D, row/column codes and scales all had 0% differing bytes.
- Midpoint-sensitive PyTorch-oracle test: 2 passed, 148 deselected.
- Grouped E=4 TE-derived oracle tests: 4 passed.
- Actual grouped TE conformance: E=1 grouped matched TE single exactly; at E=4 both TorchAO backends had 0% differing bytes from grouped TE across row/column codes and scales.
- Ruff was unavailable, and this checkout has no `.lintrunner.toml`; no package installation was attempted.

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

Review and commit the bounded TE-default arithmetic/oracle change. The full expensive matrix is intentionally deferred until pre-merge validation.

## Risk

LOW-MEDIUM: targeted TE, midpoint, grouped, and performance validation passed. The intended performance regression from precise division is measured; repository lint tooling was unavailable in this checkout.
