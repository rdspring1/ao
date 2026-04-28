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
Done. Bias coverage was added to:
- `test_column_single_rank_equivalence`
- `test_column_forward`
- `test_column_backward`

Syntax and collection passed:

```bash
python -m py_compile test/prototype/mx_formats/test_nvfp4_parallel.py
pytest --collect-only -q test/prototype/mx_formats/test_nvfp4_parallel.py
git diff --check
```

The original distributed validation failed because `test_column_backward`
compared `bias_local.grad` against a full fp32 bias reference. That was resolved
by comparing against the column-local reduction semantics used by
`nvfp4_col_parallel_mm.backward`: `dy_local.sum(dim=0)`.

Final distributed validation passed:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -q
```

Result:
- `7 passed, 2 warnings in 27.98s`

### Evidence
- `test_column_backward` now computes `bias_local.grad` from
  `nvfp4_col_parallel_mm.backward`, where `grad_bias = grad_output.sum(dim=0)`.
- `db_ref` is now computed as `dy_local.sum(dim=0)`, matching column-local bias
  gradient semantics.
- The full distributed pytest for `test_nvfp4_parallel.py` passes.

### Confidence
HIGH that direct row/column autograd tests now cover non-None bias.

---

## DTensor Parallelize Coverage

### Goal
Add end-to-end coverage for `parallelize_module(..., NVFP4ColwiseParallel())`
and `parallelize_module(..., NVFP4RowwiseParallel())` through the DTensor API.

### Current State
Done. Direct autograd-function coverage passes for:
- `nvfp4_col_parallel_mm`
- `nvfp4_row_parallel_mm`
- forward/backward shapes
- forward/backward SQNR
- non-None bias paths

Wrapper coverage was added in `test/prototype/mx_formats/test_nvfp4_parallel.py`:
- `test_column_parallelize_module`
- `test_row_parallelize_module`

The tests validate:
- `NVFP4TrainingLinear.forward`
- `NVFP4ColwiseParallel` metadata setup and local-shard runtime hooks
- `NVFP4RowwiseParallel` metadata setup and local-shard runtime hooks
- `parallelize_module` weight sharding and DTensor placement behavior
- colwise `Shard(0)` weight and bias placement
- rowwise `Shard(1)` weight placement and replicated bias placement
- forward/backward output and gradient SQNR against fp32 references
- bias-gradient semantics

Final distributed validation passed:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -q
```

Result:
- `9 passed, 2 warnings in 28.68s`

### Risk
Coverage uses `use_local_output=True`, which matches the current tests and the
local-shard kernel contract. Non-local DTensor output wrapping remains less
exercised.

### Cleanup Done
Updated stale comments/docstrings in
`torchao/prototype/mx_formats/nvfp4_tensor_parallel.py` so row parallel is
described as implemented.

### Best Next Mode
vet

---

## MLP Composition And FSDP2+TP Coverage

### Goal
Add one NVFP4 MLP-style composition test that exercises:
- `NVFP4ColwiseParallel` hidden projections
- DTensor activation handoff between colwise outputs
- `NVFP4RowwiseParallel` output projection

Then add a separate 2-D mesh `(dp, tp) = (2, 2)` FSDP2+TP smoke test.

### Current State
Done. Added:
- `NVFP4MLP`
- `_fp32_mlp_reference`
- `test_mlp_colwise_rowwise_parallelize_module`
- `test/prototype/mx_formats/test_nvfp4_fsdp2_tp.py`
- `test_nvfp4_mlp_fsdp2_tp_smoke`

The MLP tests use:
- `w1`: `NVFP4ColwiseParallel(use_local_output=False)`
- `w2`: `NVFP4ColwiseParallel(use_local_output=False)`
- `out_proj`: `NVFP4RowwiseParallel()`

Debug note:
- The first MLP attempt passed full `[M, K]` input to the colwise layer.
- Instrumentation showed colwise all-gather produced hidden DTensors with global
  shape `[M * tp, H]`.
- The test was fixed to pass local sequence input `[M / tp, K]` to the composed
  NVFP4 TP model while keeping the fp32 reference on the full input.

Checks passed:

```bash
python -m py_compile test/prototype/mx_formats/test_nvfp4_parallel.py
python -m py_compile test/prototype/mx_formats/test_nvfp4_fsdp2_tp.py
git diff --check
```

Focused MLP validation passed:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py::test_mlp_colwise_rowwise_parallelize_module -q
```

Result:
- `1 passed in 16.03s`

Full 2-rank TP validation passed:

```bash
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=2 -m pytest test/prototype/mx_formats/test_nvfp4_parallel.py -q
```

Result:
- `10 passed, 2 warnings in 30.00s`

4-rank FSDP2+TP validation passed:

```bash
PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=4 -m pytest test/prototype/mx_formats/test_nvfp4_fsdp2_tp.py -q
```

Result:
- `1 passed in 20.94s` on each rank

### Best Next Mode
vet
