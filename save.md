# Save Note

## Goal
Make `test/prototype/mx_formats/test_nvfp4_col_parallel.py` runnable, add
backward numerical correctness coverage, then run:

```bash
torchrun --nproc_per_node=2 test/prototype/mx_formats/test_nvfp4_col_parallel.py
```

## Current State
Done. The 2-rank torchrun test passes on branch `tp`.

Final validation:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 test/prototype/mx_formats/test_nvfp4_col_parallel.py
```

Result:
- `_test_swap_first_dims`: PASSED
- `_test_single_rank_equivalence`: PASSED
- `_test_forward_shapes`: PASSED
- `_test_backward_shapes`: PASSED with `dx SQNR=15.17 dB`, `dw SQNR=15.55 dB`
- `_test_forward_numerics`: PASSED with `SQNR=16.62 dB`
- Final output: `All tests passed.`

Focused checks also passed:

```bash
python -m py_compile torchao/prototype/mx_formats/nvfp4_tensor_parallel.py test/prototype/mx_formats/test_nvfp4_col_parallel.py
git diff --check
```

## What Changed
- Added CUDA graph allocator setup to `_triton_rht_quantize_from_amax`, because
  tensor-parallel code calls the helper directly and otherwise skips the public
  wrapper's `prepare_for_cuda_graph` path.
- Updated `NVFP4ColwiseParallel` runtime hooks to pass local input/output shards
  through instead of letting parent `ColwiseParallel` all-gather BF16 inputs.
- Removed stale `compute_rowwise=True` keywords from `triton_rht_amax` calls; this
  branch's `triton_rht_amax` already returns both column and row amax tensors.
- Made `_test_swap_first_dims` all-gather inputs contiguous before calling
  `dist.all_gather`.
- Fixed `_test_single_rank_equivalence` so all ranks collectively create the rank-0
  process group before non-rank-0 returns.
- Changed the forward numerics check from elementwise `assert_close` to the repo's
  existing NVFP4 SQNR style with a 15 dB threshold.
- Added backward numerical correctness to `_test_backward_shapes`: local `dx` and
  `dw` now compare against full fp32 matmul reference gradients with 14 dB SQNR
  thresholds.
- Fixed TP wgrad correctness by making all ranks use the same deterministic RHT
  sign vector for columnwise-RHT activation and gradient quantization.

## Evidence
- Initial backward numerics attempt found `dw SQNR ~= 1 dB`, while single-GPU
  `nvfp4_mm_triton` backward got `dx ~= 15.1 dB`, `dw ~= 15.5 dB` versus fp32.
- Root cause: TP gathered RHT-transformed `x` shards from ranks that had used
  different per-device random RHT sign vectors. Wgrad requires gathered `x_col`
  and local `dy_col` to share the same RHT basis.
- After adding a shared TP sign vector, the distributed test reports
  `dx SQNR=15.17 dB`, `dw SQNR=15.55 dB`.

## Confidence
HIGH that direct `nvfp4_col_parallel_mm` forward/backward works on the available
2-GPU SM100 setup.

## Risk
The test still does not validate `parallelize_module(..., NVFP4ColwiseParallel())`
end to end through the DTensor API.

## Best Next Mode
vet

---

## Bias Test Handoff

### Goal
Update `test/prototype/mx_formats/test_nvfp4_parallel.py` column-parallel tests to
exercise non-None bias in forward and backward.

### Current State
Implementation attempt added bias to:
- `test_column_single_rank_equivalence`
- `test_column_forward`
- `test_column_backward`

Syntax and collection passed:

```bash
python -m py_compile test/prototype/mx_formats/test_nvfp4_parallel.py
pytest --collect-only -q test/prototype/mx_formats/test_nvfp4_parallel.py
git diff --check
```

First distributed validation failed:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -q
```

Failure: `test_column_backward` only. The new assertion
`torch.testing.assert_close(bias_local.grad.float(), db_ref)` mismatches the
fp32 reference by about `0.12-0.15` absolute / `0.0038` relative. The existing
`dx` and `dw` SQNR assertions passed before this assertion.

### Evidence
- `test_column_backward` now computes `bias_local.grad` from
  `nvfp4_col_parallel_mm.backward`, where `grad_bias = grad_output.sum(dim=0)`.
- `db_ref` is currently computed by a full fp32 reference graph:
  `bias_ref = bias_full.float().detach().requires_grad_(True)`.
- Failure suggests the test is likely comparing a BF16/CUDA reduction result
  against fp32 accumulation too strictly, not exposing a column TP communication
  issue.

### One Concrete Next Action
Switch the bias-gradient check to compare against the same local reduction
semantics as the implementation, e.g. `dy_local.sum(dim=0)`, or use a tolerance
appropriate for BF16 reduction.

### Expected Outcome
Distributed column pytest should pass with bias covered in single-rank,
forward, and backward tests.

### Confidence
MEDIUM.

### Risk
Bias gradient semantics for TP are local-column only in column parallel. The next
check should avoid accidentally testing unrelated fp32 accumulation behavior.

### Best Next Mode
debug
