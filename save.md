# Save Note

## Goal
Make `nvfp4_mm_triton` backward compatible with `torch.compile(fullgraph=True)` via compiled autograd, with verified SR diversity across replays.

## Regression Status (all green)
- `test_nvfp4_mm_triton_cuda_graph_compile` — PASSED
- `test_triton_rht_quantize_row_col_cuda_graph_compile` — PASSED

## Applied Changes (in tree)
- `hadamard_utils.py`: added `_SR_SEED_BUFS`/`_SR_OFFSET_BUFS` dicts, `get_sr_buffers()`, pre-alloc in `prepare_for_cuda_graph()`
- `nvfp4_linear.py`: `.random_()` → `get_sr_buffers() + torch.randint + copy_ + add_(1)`
- `test_nvfp4_tensor.py`: added `test_nvfp4_mm_triton_backward_sr_diversity_compiled_backward`

## Outstanding Failure
`test_nvfp4_mm_triton_backward_sr_diversity_compiled_backward` fails on first forward call:
```
hadamard_amax_triton.py:180 → get_rht_matrix(...).to(bfloat16)
→ FunctionalTensor.to() → AttributeError: 'NoneType' object has no attribute 'export'
```

### Root Cause Chain
1. `@torch.compile(fullgraph=True)` on `fwd` triggers AOT autograd
2. AOT autograd joint-traces forward + backward in functional dispatch mode
3. `triton_rht_quantize_row_col` has `mutates_args=("seed_base",)` → `adinplaceorview` dispatch executes real backend during tracing
4. Backend calls `triton_rht_amax` → `get_rht_matrix(...).to(bfloat16)` on `lru_cache` real tensor
5. Under functional dispatch mode, `.to()` dereferences `_detect_infra_mode(FUNCTIONAL).export` → `None` → crash

## Three Approaches to Fix

| Approach | Description | Risk |
|---|---|---|
| A | Wrap `triton_rht_amax` as a custom_op with `register_fake` so AOT doesn't trace into it | Medium — requires fake implementation |
| B | In `triton_rht_amax`, detach/clone the RHT matrix before `.to()`: `get_rht_matrix(...).detach().to(bfloat16)` | Low — minimal change, may sidestep FunctionalTensor |
| C | Pre-compute the bfloat16 matrix in `get_rht_matrix` or cache the converted version so `.to()` is never called during dispatch | Low — purely additive |

## Best Next Mode
Debugger — confirm which approach unblocks the FunctionalTensor issue in `hadamard_amax_triton.py:180`.
