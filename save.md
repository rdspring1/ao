# Save Note

## Goal
Make `nvfp4_mm_triton` backward compatible with `torch.compile(fullgraph=True)` via compiled autograd, with verified SR diversity (fixed per-layer seed, monotonically increasing offset).

## Regression Status
- `test_nvfp4_mm_triton_cuda_graph_compile` — UNKNOWN (blocked before reaching it)
- `test_triton_rht_quantize_row_col_cuda_graph_compile` — UNKNOWN (blocked before reaching it)
- `test_nvfp4_mm_triton_backward_sr_diversity_compiled_backward` — FAILING

## Current State of Code (in tree)

### `hadamard_quantize_row_col_triton.py`
- `mutates_args=()` (was `mutates_args=("seed_base",)`) — DONE

### `hadamard_utils.py`
- `_device_key()` normalization added — DONE
- `_SR_SEED_BUFS`, `_SR_OFFSET_BUFS`, `get_sr_buffers()` removed — DONE

### `nvfp4_linear.py`
- `nvfp4_mm_triton.forward` signature: `(ctx, input_hp, weight_hp, bias, kernel_preference, sr_seed, sr_offset)` — 6 tensor args
- `sr_offset` = per-layer monotonic counter for backward SR (saved via `save_for_backward`)
- `sr_offset.add_(1)` at END of backward (after `ctx.saved_tensors` access → no version mismatch)
- Backward returns 6 values: `grad_input, grad_weight, grad_bias, None, None, None`
- `Nvfp4Linear`: has `sr_seed` (fixed) + `sr_offset` (starts at 0) registered buffers; no add_ in forward
- `nvfp4_linear()` wrapper: accepts optional `sr_seed`/`sr_offset`, creates zeros if None

### `test_nvfp4_tensor.py`
- `test_nvfp4_mm_triton_backward_sr_diversity_compiled_backward` uses `Nvfp4Linear`, `torch._dynamo.compiled_autograd._enable`

## Outstanding Failure

`RuntimeError: Detected 2 tensor(s) in the cudagraph pool not tracked as outputs`

Location: `cudagraph_trees.py:1956` — fires during compiled autograd + reduce-overhead CUDA graph capture.

### Key Evidence
- Error count is exactly 2 across ALL variants tested (with/without saving sr buffers, with/without cloning, with/without renaming)
- Pre-dates any sr_offset changes — unrelated to SR state management
- Only fires with compiled autograd + `mode="reduce-overhead"` (CUDA graphs)
- Error message explicitly suggests: "Set `torch._inductor.config.triton.cudagraph_trees_history_recording = True` for allocation origins"

### Root Cause (unknown)
The 2 leaked tensors are allocated inside the compiled backward CUDA graph and held by some Python reference that persists after the graph completes. Most likely candidates:
1. `_, _, _` discards from `triton_rht_quantize_row_col` in GEMM3 backward — 6 outputs, 3 discarded
2. Internal compiled autograd SavedVariable wrappers for `sr_seed`/`sr_offset`
3. Some intermediate in the backward graph

## Next Action
Run with history recording to identify the 2 allocation sites:
```python
import torch._inductor.config
torch._inductor.config.triton.cudagraph_trees_history_recording = True
```
Then run the test and read the "history:" field in the error message.

Command:
```bash
pytest test/prototype/mx_formats/test_nvfp4_tensor.py::test_nvfp4_mm_triton_backward_sr_diversity_compiled_backward -xvs
```

## Best Next Mode
Debugger — identify the 2 leaked allocation sites, then fix.

## Confidence: MEDIUM (SR logic is correct; CUDA graph pool issue is external to SR changes)
## Risk: LOW (changes so far are minimal and reversible)
