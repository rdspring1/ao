# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 training linear layer with quantized forward and backward passes.

Modeled on mx_linear.py (MXFP8 training), this implements an autograd function
that quantizes all three GEMMs in a Linear layer to NVFP4:

    Forward:  input @ weight^T = output
    Backward: grad_output @ weight = grad_input
    Backward: input^T @ grad_output = grad_weight

The quantization step is pluggable: by default uses TorchAO's NVFP4Tensor, but
can optionally use TransformerEngine's NVFP4Quantizer for features not yet in
TorchAO (RHT, stochastic rounding). As those features land upstream, the TE
backend can be removed.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.custom_fp_utils import RoundingMode
from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)
from torchao.prototype.mx_formats.kernels import triton_quantize_nvfp4
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.quantize_2d_triton import triton_weight_quantize_2d
from torchao.prototype.mx_formats.utils import (
    hp_data_dims_to_swizzled_scale_dims_nvfp4,
    to_blocked,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


def _quantize_to_nvfp4(
    tensor: torch.Tensor,
    *,
    kernel_preference: KernelPreference = KernelPreference.TORCH,
    stochastic_rounding: bool = False,
    random_hadamard_transform: bool = False,
) -> NVFP4Tensor:
    """Quantize a high-precision tensor to NVFP4.

    Pluggable backend: uses TorchAO by default, optionally TE for features
    not yet upstream (stochastic rounding, RHT).

    Args:
        tensor: 2D high-precision tensor (bf16 or fp32)
        kernel_preference: Backend to use for quantization. TORCH uses TorchAO's
            native path. TE uses TransformerEngine's NVFP4Quantizer (enables
            stochastic rounding and RHT). AUTO selects TE if available, else TORCH.
        stochastic_rounding: Enable stochastic rounding (TE backend only for now)
        random_hadamard_transform: Enable RHT (TE backend only for now)
    """
    effective = kernel_preference
    if effective == KernelPreference.AUTO:
        try:
            import transformer_engine  # noqa: F401
            effective = KernelPreference.TE
        except ImportError:
            effective = KernelPreference.TORCH

    if effective == KernelPreference.TE:
        return _quantize_to_nvfp4_te(
            tensor,
            stochastic_rounding=stochastic_rounding,
            random_hadamard_transform=random_hadamard_transform,
        )

    # TorchAO path: standard NVFP4 quantization (round-to-nearest, no RHT)
    tensor_amax = torch.max(torch.abs(tensor))
    per_tensor_scale = per_tensor_amax_to_scale(tensor_amax)
    return NVFP4Tensor.to_nvfp4(
        tensor,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
    )


def _quantize_to_nvfp4_te(
    tensor: torch.Tensor,
    *,
    stochastic_rounding: bool = False,
    random_hadamard_transform: bool = False,
) -> NVFP4Tensor:
    """Quantize using TransformerEngine's NVFP4Quantizer, returning a TorchAO NVFP4Tensor.

    Bridges TE's quantization output (which supports RHT and stochastic rounding)
    into TorchAO's NVFP4Tensor format for use with torch._scaled_mm via
    _addmm_nvfp4_dispatch().

    Args:
        tensor: 2D high-precision tensor (bf16 or fp32)
        stochastic_rounding: Enable stochastic rounding (for gradients)
        random_hadamard_transform: Enable Random Hadamard Transform (for activations/gradients)
    """
    from transformer_engine.pytorch.tensor import NVFP4Quantizer

    te_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_rht=random_hadamard_transform,
        with_post_rht_amax=random_hadamard_transform,
        stochastic_rounding=stochastic_rounding,
    )

    # TE's RHT requires bf16 input
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)

    te_out = te_quantizer(tensor)

    # Extract TE components: packed FP4 data and E4M3 block scales
    te_qdata = te_out._rowwise_data           # uint8, (M, K//2)
    te_scales_raw = te_out._rowwise_scale_inv.view(torch.float8_e4m3fn)
    te_amax = te_out._amax_rowwise            # float32, (1,)

    # Convert amax to TorchAO's per_tensor_scale format
    per_tensor_scale = per_tensor_amax_to_scale(te_amax)

    # TE pads the scale M-dimension to next multiple of 128.
    # Slice off the padding to get (M, K//16), then swizzle for torch._scaled_mm.
    M, K = tensor.shape
    te_scales = te_scales_raw[:M, :K // 16]
    te_scales_swizzled = to_blocked(te_scales.contiguous()).flatten()
    scale_M, scale_K = hp_data_dims_to_swizzled_scale_dims_nvfp4(M, K)
    te_scales_reshaped = te_scales_swizzled.view(scale_M, scale_K)

    return NVFP4Tensor(
        te_qdata,
        te_scales_reshaped,
        16,
        tensor.dtype,
        per_tensor_scale=per_tensor_scale,
        is_swizzled_scales=True,
    )


def _ao_rowwise_quantize_sr(x: torch.Tensor):
    """Triton NVFP4 rowwise quantization with stochastic rounding.

    Returns (fp4_data, block_scales, global_scale).
    block_scales: (M, N//16) float8_e4m3fn in SWIZZLE_32_4_4 memory layout.
    """
    global_scale = per_tensor_amax_to_scale(x.abs().max())
    seed = torch.randint(0, 2**31 - 1, (1,), dtype=torch.int32, device=x.device)
    scales, xq = triton_quantize_nvfp4(x, global_scale, RoundingMode.RS.value, seed)
    return xq.view(torch.float4_e2m1fn_x2), scales, global_scale


def _triton_weight_quantize_2d(x: torch.Tensor):
    """Triton 2D (16×16) NVFP4 weight quantization.

    Returns (fp4_data, block_scales_flat, global_scale).
    block_scales_flat: swizzled scales flattened for scaled_mm.
    """
    codes, sf, global_amax = triton_weight_quantize_2d(x, swizzle_scale_factors=True)
    return (
        codes.view(torch.float4_e2m1fn_x2),
        sf.flatten(),
        per_tensor_amax_to_scale(global_amax),
    )


def _triton_weight_quantize_2d_colwise(x: torch.Tensor):
    """Triton 2D NVFP4 weight quantization producing both rowwise and colwise outputs.

    Returns (W_fp4_x2, W_bs, W_gs, Wt_fp4_x2, Wt_sf, W_amax) where:
      W_*  = rowwise quantized x (for forward GEMM)
      Wt_* = colwise quantized x = rowwise quantized x.T (for dgrad GEMM)
    """
    codes, sf, t_codes, t_sf, global_amax = triton_weight_quantize_2d(
        x, compute_colwise=True
    )
    return (
        codes.view(torch.float4_e2m1fn_x2),
        sf.flatten(),
        per_tensor_amax_to_scale(global_amax),
        t_codes.view(torch.float4_e2m1fn_x2),
        t_sf,
        global_amax,
    )


@torch._dynamo.allow_in_graph
class nvfp4_mm_triton(torch.autograd.Function):
    """NVFP4 quantized matmul: pure-triton RHT + stochastic rounding path.

    3 GEMMs:
      forward:   x_row @ W.T  = output         (triton RHT rowwise + 2D weight)
      backward:  dy_sr @ W.T  = grad_input      (triton SR rowwise + 2D weight)
      backward:  dy_col.T @ x_col = grad_weight (triton col RHT + SR for dy; saved col for x)

    Requires: bfloat16 input, M % 128 == 0, K % 128 == 0, N % 128 == 0.
    Saves only FP4 codes+scales for backward (memory efficient vs full-precision activations).
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        kernel_preference: KernelPreference,
    ):
        M = input_hp.shape[-2]
        K = input_hp.shape[-1]
        N = weight_hp.shape[0]
        if input_hp.dtype != torch.bfloat16:
            input_hp = input_hp.to(torch.bfloat16)
        if weight_hp.dtype != torch.bfloat16:
            weight_hp = weight_hp.to(torch.bfloat16)
        if M % 128 != 0 or K % 128 != 0 or N % 128 != 0:
            raise ValueError(
                f"nvfp4_mm_triton requires M, K, N all divisible by 128; "
                f"got M={M}, K={K}, N={N}"
            )
        input_2d = input_hp.reshape(-1, K)

        # RHT + columnwise + rowwise quantization of input in one fused kernel
        (
            x_col_codes, x_col_sf, x_col_amax,
            x_row_codes, x_row_sf, x_row_amax,
        ) = triton_rht_quantize_row_col(
            input_2d.contiguous(), stochastic_rounding=False, compute_rowwise=True
        )

        # Fused weight quantization: rowwise for forward GEMM, colwise saved for dgrad
        W_fp4_x2, W_bs, W_gs, Wt_fp4_x2, Wt_sf, W_amax = _triton_weight_quantize_2d_colwise(weight_hp)
        x_gs = per_tensor_amax_to_scale(x_row_amax)

        output = torch.nn.functional.scaled_mm(
            x_row_codes.view(torch.float4_e2m1fn_x2),
            W_fp4_x2.t(),
            scale_a=[x_row_sf.flatten(), x_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[W_bs, W_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        output = output.reshape(*input_hp.shape[:-1], N)
        if bias is not None:
            output = output + bias

        # Save columnwise x and pre-quantized W.T for backward (FP4 — memory efficient)
        ctx.save_for_backward(x_col_codes, x_col_sf, x_col_amax, Wt_fp4_x2, Wt_sf, W_amax)
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x_col_codes, x_col_sf, x_col_amax, Wt_fp4_x2, Wt_sf, W_amax = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        # -----------------------------------------------------------
        # GEMM 2: dy_sr @ W.T → grad_input  (SR rowwise; saved colwise W)
        # -----------------------------------------------------------
        dy_fp4, dy_bs, dy_gs = _ao_rowwise_quantize_sr(grad_output_2d)
        Wt_bs = Wt_sf.flatten()
        Wt_gs = per_tensor_amax_to_scale(W_amax)
        grad_input = torch.nn.functional.scaled_mm(
            dy_fp4,
            Wt_fp4_x2.t(),
            scale_a=[dy_bs, dy_gs],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[Wt_bs, Wt_gs],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )
        grad_input = grad_input.reshape(ctx.input_orig_shape)

        # -----------------------------------------------------------
        # GEMM 3: dy_col.T @ x_col → grad_weight  (col RHT + SR)
        # -----------------------------------------------------------
        # Both col scale tensors are in SWIZZLE_32_4_4 layout — just flatten.
        dy_col_codes, dy_col_sf, dy_col_amax, _, _, _ = triton_rht_quantize_row_col(
            grad_output_2d, stochastic_rounding=True, compute_rowwise=False
        )
        dy_gs_w = per_tensor_amax_to_scale(dy_col_amax)
        x_gs_w = per_tensor_amax_to_scale(x_col_amax)
        grad_weight = torch.nn.functional.scaled_mm(
            dy_col_codes.view(torch.float4_e2m1fn_x2),
            x_col_codes.view(torch.float4_e2m1fn_x2).t(),
            scale_a=[dy_col_sf.flatten(), dy_gs_w],
            scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            scale_b=[x_col_sf.flatten(), x_gs_w],
            scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
            swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
            output_dtype=torch.bfloat16,
        )

        grad_bias = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias else None
        )
        return grad_input, grad_weight, grad_bias, None


def nvfp4_linear(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    kernel_preference: KernelPreference = KernelPreference.TORCH,
) -> torch.Tensor:
    """Convenience wrapper around the nvfp4_mm autograd function.

    Performs a quantized linear operation: output = input @ weight^T + bias,
    with NVFP4 quantization on forward and backward GEMMs.

    Args:
        input_hp: High precision input [..., in_features]
        weight_hp: High precision weight [out_features, in_features]
        bias: Optional bias [out_features]
        kernel_preference: Backend for quantization (TORCH, TE, AUTO, or TRITON)
    """
    if kernel_preference == KernelPreference.TRITON:
        return nvfp4_mm_triton.apply(input_hp, weight_hp, bias, kernel_preference)
    return nvfp4_mm.apply(input_hp, weight_hp, bias, kernel_preference)


@torch._dynamo.allow_in_graph
class nvfp4_mm(torch.autograd.Function):
    """NVFP4 quantized matmul for training.

    Three GEMMs in a Linear forward + backward, all in NVFP4:

    1. Forward:  input @ weight^T    = output      (both → FP4)
    2. Backward: grad_output @ weight = grad_input  (both → FP4)
    3. Backward: input^T @ grad_output = grad_weight (both → FP4)

    Per the NVFP4 training recipe (see NVFP4BlockScaling):
    - Forward activations: RHT applied before quantization
    - Forward weights: 2D block quantization (16x16)
    - Backward gradients: RHT + stochastic rounding before quantization

    With kernel_preference=TORCH, uses TorchAO's round-to-nearest quantization
    (no RHT or stochastic rounding). With kernel_preference=TE, uses
    TransformerEngine's NVFP4Quantizer with the full recipe (RHT on activations
    and gradients, stochastic rounding on gradients).
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        kernel_preference: KernelPreference,
    ):
        # input @ weight^T = output
        input_orig_shape = input_hp.shape
        input_hp_2d = input_hp.reshape(-1, input_orig_shape[-1])

        # Quantize input activations — RHT applied for TE backend
        use_te = kernel_preference == KernelPreference.TE or (
            kernel_preference == KernelPreference.AUTO
        )
        input_nvfp4 = _quantize_to_nvfp4(
            input_hp_2d,
            kernel_preference=kernel_preference,
            random_hadamard_transform=use_te,
        )

        # Quantize weights (2D block quant would be applied here)
        weight_nvfp4 = _quantize_to_nvfp4(
            weight_hp,
            kernel_preference=kernel_preference,
        )

        output = _addmm_nvfp4_dispatch(
            input_nvfp4, weight_nvfp4.t(), None, bias=bias
        )
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

        # Save quantized input components (FP4) instead of full-precision (bf16)
        # to reduce activation memory ~4x. Weight is the parameter itself (no extra copy).
        ctx.save_for_backward(
            input_nvfp4.qdata, input_nvfp4.scale, input_nvfp4.per_tensor_scale,
            weight_hp,
        )
        ctx.input_orig_shape = input_orig_shape
        ctx.input_nvfp4_is_swizzled = input_nvfp4.is_swizzled_scales
        ctx.input_nvfp4_orig_dtype = input_nvfp4.orig_dtype
        ctx.kernel_preference = kernel_preference
        ctx.use_te = use_te
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def backward(ctx, grad_output_hp: torch.Tensor):
        input_qdata, input_scale, input_per_tensor_scale, weight_hp = ctx.saved_tensors
        kernel_preference = ctx.kernel_preference
        use_te = ctx.use_te

        grad_output_orig_shape = grad_output_hp.shape
        grad_output_2d = grad_output_hp.reshape(-1, grad_output_orig_shape[-1])

        # -----------------------------------------------------------
        # GEMM 2: grad_output @ weight = grad_input
        # -----------------------------------------------------------
        # Quantize gradient — stochastic rounding + RHT for TE backend
        grad_nvfp4 = _quantize_to_nvfp4(
            grad_output_2d,
            kernel_preference=kernel_preference,
            stochastic_rounding=use_te,
            random_hadamard_transform=use_te,
        )

        # Quantize weight for backward (no RHT, no stochastic rounding)
        weight_nvfp4 = _quantize_to_nvfp4(
            weight_hp.t().contiguous(),
            kernel_preference=kernel_preference,
        )

        grad_input = _addmm_nvfp4_dispatch(grad_nvfp4, weight_nvfp4.t(), None)
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        # -----------------------------------------------------------
        # GEMM 3: grad_output^T @ input = grad_weight
        # -----------------------------------------------------------
        # Quantize grad_output along the other dimension — SR + RHT for TE backend
        grad_t_nvfp4 = _quantize_to_nvfp4(
            grad_output_2d.t().contiguous(),
            kernel_preference=kernel_preference,
            stochastic_rounding=use_te,
            random_hadamard_transform=use_te,
        )

        # Reconstruct quantized input from saved FP4 components, dequantize,
        # transpose, and re-quantize for GEMM 3 (which needs the transposed layout).
        # Still a net memory win: we stored FP4 (~0.5 bytes/elem) instead of bf16 (2 bytes/elem).
        input_nvfp4 = NVFP4Tensor(
            input_qdata, input_scale, 16, ctx.input_nvfp4_orig_dtype,
            per_tensor_scale=input_per_tensor_scale,
            is_swizzled_scales=ctx.input_nvfp4_is_swizzled,
        )
        input_hp_2d = input_nvfp4.dequantize(ctx.input_nvfp4_orig_dtype)
        input_t_nvfp4 = _quantize_to_nvfp4(
            input_hp_2d.t().contiguous(),
            kernel_preference=kernel_preference,
        )

        grad_weight = _addmm_nvfp4_dispatch(
            grad_t_nvfp4, input_t_nvfp4.t(), None
        )

        # Bias gradient is just sum of grad_output along batch dims
        grad_bias = grad_output_hp.sum(dim=tuple(range(grad_output_hp.dim() - 1))) if ctx.has_bias else None

        return grad_input, grad_weight, grad_bias, None
