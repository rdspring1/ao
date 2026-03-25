# NVFP4 Training: Stochastic Rounding + Hadamard Transform

## Overview

Add training support for NVFP4 via a custom autograd function, mirroring the
`mx_linear.py` pattern. Key features: per-direction (fwd/dgrad/wgrad) control
over stochastic rounding (SR) and Random Hadamard Transform (RHT).

## Background

`NVFP4Tensor` + `F.linear` is inference-only (`requires_grad=False` hardcoded).
Training requires:
- HP tensors saved for backward
- Different quantization config per matmul direction
- RHT applied to wgrad inputs only (current paradigm)

Analogous to `mx_linear.py:mx_mm`, which handles MXFP8 training with
per-direction quantization for fwd/dgrad/wgrad.

### RHT scope
RHT is currently only applied to the wgrad matmul inputs (`input_t` and
`grad_output`). Applying RHT to fwd or dgrad would require independent sign
vectors for each direction — that is out of scope for now. The kwarg name
`rht_sign_vector` is intentionally general to allow future extension, but the
autograd function will guard that fwd and dgrad kwargs leave it unset.

## New Files

### `torchao/prototype/mx_formats/nvfp4_linear.py`
Mirrors `mx_linear.py`. Contains:

- **`NVFP4TrainingConfig`** dataclass — holds per-direction
  `QuantizeTensorToNVFP4Kwargs` for fwd, dgrad, wgrad:
  ```python
  @dataclass
  class NVFP4TrainingConfig:
      fwd_kwargs:   QuantizeTensorToNVFP4Kwargs = field(default_factory=...)
      dgrad_kwargs: QuantizeTensorToNVFP4Kwargs = field(default_factory=...)
      wgrad_kwargs: QuantizeTensorToNVFP4Kwargs = field(default_factory=...)
  ```

- **`nvfp4_mm`** (`torch.autograd.Function`) — three GEMMs:
  ```
  forward:  input @ weight.T  = output      (fwd_kwargs)
  backward: grad_out @ weight = grad_input  (dgrad_kwargs)
  backward: input.T @ grad_out = grad_weight (wgrad_kwargs, RHT applied here only)
  ```
  Guards at entry:
  ```python
  assert fwd_kwargs.rht_sign_vector is None, "RHT for fwd not yet supported"
  assert dgrad_kwargs.rht_sign_vector is None, "RHT for dgrad not yet supported"
  ```

- **`nvfp4_linear`** — convenience wrapper (mirrors `_to_mxfp8_then_scaled_mm`)

### `QuantizeTensorToNVFP4Kwargs` changes (existing file)
Add one new field:
```python
@dataclass
class QuantizeTensorToNVFP4Kwargs(QuantizeTensorKwargs):
    ...
    use_stochastic_rounding: bool = False   # NEW
    rht_sign_vector: Optional[torch.Tensor] = None  # already exists at tensor level, promote here
```
`rht_sign_vector` already exists on `NVFP4Tensor` for inference use. Promoting
it into `QuantizeTensorToNVFP4Kwargs` makes it settable per-direction in
training. The name is kept general — the restriction to wgrad-only is enforced
by the guards in `nvfp4_mm`, not by the kwarg name.

## Modified Files

| File | Change |
|------|--------|
| `torchao/prototype/mx_formats/nvfp4_linear.py` | **new** — autograd function + config |
| `torchao/prototype/mx_formats/config.py` | add `NVFP4TrainingConfig` if shared config lives here |
| `torchao/prototype/mx_formats/__init__.py` | export new symbols |
| `torchao/prototype/mx_formats/kernels.py` | add SR path to quantization kernels (PR 1) |

## Tests

### `test/prototype/mx_formats/test_kernels.py`
Unit tests for SR kernel — same pattern as `test_triton_mxfp8_dim0_randn`:
- SR kernel output vs reference: **bitwise equality** with fixed seed
- Verify SR output differs from round-to-nearest (statistical test)

### `test/prototype/mx_formats/test_nvfp4_linear.py` (new)
Mirrors `test/prototype/mx_formats/test_mx_mm.py`. Tests for the autograd function:
- **Correctness**: fwd output SQNR vs BF16 reference (≥ 16 dB)
- **Gradient flow**: `grad_input` and `grad_weight` are not None, correct shape
- **RHT**: fwd + inverse RHT in dgrad produces higher SQNR than without
- **SR**: output differs across calls with same input (non-determinism check)
- **Config combinations**: parametrize over fwd/dgrad/wgrad SR and RHT on/off

### `test/prototype/mx_formats/test_nvfp4_tensor.py`
- Add `use_stochastic_rounding` to `QuantizeTensorToNVFP4Kwargs` parametrization
  in existing `test_nvfp4_matmul_with_amax`

## ghstack

```
PR 1 — SR + RHT quantization kernels (test_kernels.py)
  └── Add SR path to nvfp4 quantization kernel
  └── test_kernels.py: bitwise equality, SR non-determinism

PR 2 — QuantizeTensorToNVFP4Kwargs knobs (stacks on PR 1)
  └── Add use_stochastic_rounding to QuantizeTensorToNVFP4Kwargs
  └── Thread through NVFP4Tensor.__new__, to_nvfp4, all @implements handlers
  └── test_nvfp4_tensor.py: parametrize existing tests over new knob

PR 3 — nvfp4_linear.py autograd function (stacks on PR 2)
  └── NVFP4TrainingConfig dataclass
  └── nvfp4_mm autograd Function (fwd + dgrad + wgrad)
  └── nvfp4_linear convenience wrapper
  └── test_nvfp4_linear.py: correctness, gradients, RHT inverse, SR non-determinism
```

## Key Design Decisions

**Why not extend `NVFP4Tensor` for training?**
`requires_grad=False` is hardcoded. Training requires HP tensors saved in
`ctx`, per-direction config, and inverse RHT in backward — none of which fit
the tensor subclass model.

**Why reuse `QuantizeTensorToNVFP4Kwargs` per-direction?**
`to_nvfp4` already accepts these kwargs. Each direction in the autograd
function calls `NVFP4Tensor.to_nvfp4(..., **direction_kwargs)`, keeping the
quantization logic in one place.

**RHT scope and naming**
RHT is currently applied only to wgrad inputs. `rht_sign_vector` is named
generally in `QuantizeTensorToNVFP4Kwargs` to allow future extension to fwd
and dgrad (each of which would need its own independent sign vector). For now,
the autograd function asserts `fwd_kwargs.rht_sign_vector is None` and
`dgrad_kwargs.rht_sign_vector is None` to make the restriction explicit.
