# Save Note

## Goal
Make `nvfp4_mm_triton` fully correct under `torch.compile(mode="reduce-overhead", fullgraph=True)` CUDA graphs, including verified SR diversity in the backward pass.

## Current State
All blocking issues fixed. All CUDA graph tests pass. One remaining verification gap.

### Completed This Session

**1. `get_tma_workspace` renamed → `prepare_for_cuda_graph`**
`hadamard_utils.py`: renamed, docstring updated. All call sites in source + test files updated.

**2. `seed_buf` renamed → `seed_base`**
`hadamard_quantize_row_col_triton.py`: `mutates_args=("seed_base",)`, function param, body, `register_fake` — all updated. Stale `philox_seed_base` docstring fixed.

**3. Backward SR seeds fixed**
`nvfp4_linear.py` backward: replaced `torch.randint()` (pool-allocates, frozen) with `torch.empty() + .random_()`. Under `torch.compile(mode="reduce-overhead")` cudagraph trees, the `.random_()` CUDA RNG kernel advances each replay → SR diversity.

Key constraint: global pre-allocated tensors (via `get_sr_bufs()`) cannot be passed to `triton_rht_quantize_row_col` inside a compiled region — the custom_op's FakeTensor dispatch requires all tensor inputs to be FakeTensors, and real tensors from globals fail with `AssertionError: Please convert all Tensors to FakeTensors first`. Locally-created tensors (via `torch.empty()`) are FakeTensors during tracing.

**4. TMA allocator self-install restored**
- `hadamard_amax_triton.py`: added `prepare_for_cuda_graph` import + 3-line allocator block in `triton_rht_amax()`
- `quantize_2d_triton.py`: added allocator block in `triton_weight_quantize_2d()` (base fn, not just the wrapper)

**5. Forward `_fwd_seed_buf` / `_fwd_offset_buf` cleaned up**
`nvfp4_linear.py` forward: uses `torch.empty()` for both. SR=False, kernel ignores them. Passes as keyword args `seed_base=` / `offset_base=`.

**6. Dead code removed**
`hadamard_utils.py`: `_SR_BUFS` dict and `get_sr_bufs()` accessor removed (were unused after FakeTensor constraint was discovered).

## Test Results
```
test_triton_rht_quantize_row_col_cuda_graph_compile       PASSED
test_triton_weight_quantize_2d_colwise_cuda_graph_compile PASSED
test_nvfp4_mm_triton_cuda_graph_compile                   PASSED
130 passed (1 pre-existing unrelated MSLK failure excluded)
```

## Remaining Gap
**Backward SR diversity unverified.** No test asserts that compiled backward produces different `grad_weight` values across replays. The `.empty() + .random_()` pattern is believed correct (cudagraph trees advances CUDA RNG offset each replay) but empirically unverified.

## Next Action
Write a backward SR diversity test:
```python
# compile forward+backward under reduce-overhead
# run 3 warmup backward passes
# assert grad_weight differs between replay 4 and replay 5
```

## Why
This validates the key correctness claim. If it fails, the approach needs revision (e.g., external seed passing via function args, register_hook pattern).

## Expected Outcome
Test should pass: cudagraph trees is documented to advance CUDA RNG offset between replays.

## Confidence
HIGH (compilation, forward tests) / MEDIUM (backward SR diversity empirically unverified)

## Risk
LOW — all existing tests pass.

## Best Next Mode
Implementer — add backward SR diversity test.
