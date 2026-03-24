# MX training and inference with native PyTorch

Training documentation has moved to the mxfp8 section of [Quantized Training](https://pytorch.org/ao/main/workflows/training.html#mxfp8).

Inference documentation has moved to the the mxfp8, nvfp4 and mxfp4 sections of [Quantized Inference](https://pytorch.org/ao/main/workflows/inference.html).

## NVFP4 Features

### Random Hadamard Transform (RHT) pre-quantization

`NVFP4Tensor.from_hp` and `QuantizeTensorToNVFP4Kwargs` now accept an optional `rht_sign_vector` parameter — a ±1 tensor of length 16. When provided, a Random Hadamard Transform (`diag(sign) @ H16`) is applied to the input along the last dimension before quantization. This technique (popularized by TransformerEngine) rotates the weight distribution to reduce quantization error.

```python
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _DEFAULT_RHT_SIGN_VECTOR,
)
import torch

sign_vector = torch.tensor(_DEFAULT_RHT_SIGN_VECTOR)
q = NVFP4Tensor.from_hp(weight, rht_sign_vector=sign_vector)
```

The RHT matrix `diag(sign) @ H16` is computed once per `(sign_vector, device)` pair and cached via `functools.lru_cache`, so repeated calls are cheap.

To use the same default sign vector as TransformerEngine, pass `torch.tensor(_DEFAULT_RHT_SIGN_VECTOR)`. Pass `None` (the default) to skip the transform entirely.

### Triton quantization kernel rename

The internal Triton quantization kernel was renamed from `mslk_quantize_nvfp4` to `triton_quantize_nvfp4` for clarity. The `use_triton_kernel=True` path in `NVFP4Tensor.from_hp` also no longer requires `per_tensor_scale` to be provided; when it is `None` the kernel derives the scale from the data.

### Scale computation fix

The blockwise scale application in the pure-PyTorch path (`nvfp4_quantize`) was corrected. Previously values were scaled as `x * (1/per_tensor_scale / fp8_scale)`, which could diverge from the Triton kernel numerics. The formula is now `x / (per_tensor_scale * fp8_scale)`, which correctly inverts the combined scale.
