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
import torch.nn as nn
import torch.nn.functional as F

from torchao.prototype.custom_fp_utils import RoundingMode
from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
    triton_rht_quantize_row_col,
)
from torchao.prototype.mx_formats.hadamard_utils import (
    prepare_for_cuda_graph,
)  # noqa: F401 (re-exported for user convenience)
from torchao.prototype.mx_formats.kernels import triton_quantize_nvfp4
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    _addmm_nvfp4_dispatch,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.quantize_2d_triton import (
    triton_weight_quantize_2d,
)
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
    te_qdata = te_out._rowwise_data  # uint8, (M, K//2)
    te_scales_raw = te_out._rowwise_scale_inv.view(torch.float8_e4m3fn)
    te_amax = te_out._amax_rowwise  # float32, (1,)

    # Convert amax to TorchAO's per_tensor_scale format
    per_tensor_scale = per_tensor_amax_to_scale(te_amax)

    # TE pads the scale M-dimension to next multiple of 128.
    # Slice off the padding to get (M, K//16), then swizzle for torch._scaled_mm.
    M, K = tensor.shape
    te_scales = te_scales_raw[:M, : K // 16]
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


def _ao_rowwise_quantize_sr(
    x: torch.Tensor,
    seed_base: torch.Tensor,
    offset_base: torch.Tensor,
):
    """Triton NVFP4 rowwise quantization with stochastic rounding.

    Returns (fp4_data, block_scales, global_scale).
    block_scales: (M, N//16) float8_e4m3fn in SWIZZLE_32_4_4 memory layout.
    seed_base/offset_base are read-only; no mutation inside this function.
    """
    global_scale = per_tensor_amax_to_scale(x.abs().max())
    seed = (seed_base ^ offset_base).to(torch.int32)
    scales, xq = triton_quantize_nvfp4(x, global_scale, RoundingMode.RS.value, seed)
    return xq.view(torch.float4_e2m1fn_x2), scales, global_scale


def _triton_weight_quantize_2d(x: torch.Tensor):
    """Triton 2D NVFP4 weight quantization producing both rowwise and colwise outputs.

    Returns (W_fp4_x2, W_bs, W_gs, Wt_fp4_x2, Wt_sf, W_amax) where:
      W_*  = rowwise quantized x (for forward GEMM)
      Wt_* = colwise quantized x = rowwise quantized x.T (for dgrad GEMM)
    """
    codes, sf, t_codes, t_sf, global_amax = triton_weight_quantize_2d(x)
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

    sr_seed is a single fixed buffer giving the Philox key. Backward generates fresh
    offset_base values via torch.randint (default CUDA RNG, no generator= arg). Under
    torch.compile(mode="reduce-overhead") the default CUDA generator is a first-class
    CUDA graph side input: the framework advances it between replays, giving different
    SR noise each backward step without save_for_backward or external counter plumbing.
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp: torch.Tensor,
        bias: Optional[torch.Tensor],
        kernel_preference: KernelPreference,
        sr_seed: torch.Tensor,
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

        # RHT + columnwise + rowwise quantization of input in one fused kernel.
        # SR=False in forward — sr_seed value is not consumed here.
        (
            x_col_codes,
            x_col_sf,
            x_col_amax,
            x_row_codes,
            x_row_sf,
            x_row_amax,
        ) = triton_rht_quantize_row_col(
            input_2d.contiguous(),
            stochastic_rounding=False,
            compute_rowwise=True,
        )

        # Fused weight quantization: rowwise for forward GEMM, colwise saved for dgrad
        (
            W_fp4_x2,
            W_bs,
            W_gs,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
        ) = _triton_weight_quantize_2d(weight_hp)
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

        ctx.save_for_backward(
            x_col_codes,
            x_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        )
        ctx.input_orig_shape = input_hp.shape
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_col_codes,
            x_col_sf,
            x_col_amax,
            Wt_fp4_x2,
            Wt_sf,
            W_amax,
            sr_seed,
        ) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        dev = grad_output.device

        # Default CUDA RNG: torch.compile/reduce-overhead advances the default generator
        # between CUDA graph replays — same mechanism as dropout/randn in CUDA graphs.
        # Two independent calls give GEMM 2 and GEMM 3 different positions in the RNG stream.
        offset_rowwise = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )
        offset_colwise = torch.randint(
            -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=dev
        )

        # -----------------------------------------------------------
        # GEMM 2: dy_sr @ W.T → grad_input  (SR rowwise; saved colwise W)
        # -----------------------------------------------------------
        dy_fp4, dy_bs, dy_gs = _ao_rowwise_quantize_sr(
            grad_output_2d, sr_seed, offset_rowwise
        )
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
        dy_col_codes, dy_col_sf, dy_col_amax, _, _, _ = triton_rht_quantize_row_col(
            grad_output_2d,
            stochastic_rounding=True,
            compute_rowwise=False,
            seed_base=sr_seed,
            offset_base=offset_colwise,
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
            if ctx.has_bias
            else None
        )
        # Two extra Nones: kernel_preference, sr_seed
        return grad_input, grad_weight, grad_bias, None, None


def nvfp4_linear(
    input_hp: torch.Tensor,
    weight_hp: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    kernel_preference: KernelPreference = KernelPreference.TORCH,
    sr_seed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convenience wrapper around the nvfp4_mm autograd function.

    Performs a quantized linear operation: output = input @ weight^T + bias,
    with NVFP4 quantization on forward and backward GEMMs.

    Args:
        input_hp: High precision input [..., in_features]
        weight_hp: High precision weight [out_features, in_features]
        bias: Optional bias [out_features]
        kernel_preference: Backend for quantization (TORCH, TE, AUTO, or TRITON)
        sr_seed: Fixed int64 seed tensor (size=(1,)) for SR Philox key. Allocated
            fresh if None. For reproducibility, pass a pre-allocated module buffer.
    """
    if kernel_preference == KernelPreference.TRITON:
        if sr_seed is None:
            sr_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=input_hp.device
            )
        return nvfp4_mm_triton.apply(
            input_hp,
            weight_hp,
            bias,
            kernel_preference,
            sr_seed,
        )
    return nvfp4_mm.apply(input_hp, weight_hp, bias, kernel_preference)


class Nvfp4Linear(nn.Module):
    """NVFP4 linear layer with CUDA-graph-safe stochastic rounding.

    sr_seed is a fixed module buffer (Philox key, constant-fold by torch.compile is correct).
    Backward SR offset is generated via torch.randint (default CUDA RNG). Under
    torch.compile(mode="reduce-overhead") the default CUDA generator is a first-class
    CUDA graph side input: the framework advances it between replays, giving different
    SR noise each backward step with no external counter management.

    Training loop pattern (torch.compile):
        from torchao.prototype.mx_formats.hadamard_utils import prepare_for_cuda_graph
        layer = Nvfp4Linear(K, N).cuda()
        prepare_for_cuda_graph(layer.sr_seed.device)  # must be called before torch.compile
        compiled = torch.compile(layer, mode="reduce-overhead", fullgraph=True)
        for step in range(num_steps):
            out = compiled(x)
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, **factory_kwargs)) if bias else None
        )
        self.register_buffer(
            "sr_seed",
            torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nvfp4_mm_triton.apply(
            x,
            self.weight,
            self.bias,
            KernelPreference.TRITON,
            self.sr_seed,
        )


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

        output = _addmm_nvfp4_dispatch(input_nvfp4, weight_nvfp4.t(), None, bias=bias)
        output = output.reshape(*input_orig_shape[:-1], output.shape[-1])

        # Save quantized input components (FP4) instead of full-precision (bf16)
        # to reduce activation memory ~4x. Weight is the parameter itself (no extra copy).
        ctx.save_for_backward(
            input_nvfp4.qdata,
            input_nvfp4.scale,
            input_nvfp4.per_tensor_scale,
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
            input_qdata,
            input_scale,
            16,
            ctx.input_nvfp4_orig_dtype,
            per_tensor_scale=input_per_tensor_scale,
            is_swizzled_scales=ctx.input_nvfp4_is_swizzled,
        )
        input_hp_2d = input_nvfp4.dequantize(ctx.input_nvfp4_orig_dtype)
        input_t_nvfp4 = _quantize_to_nvfp4(
            input_hp_2d.t().contiguous(),
            kernel_preference=kernel_preference,
        )

        grad_weight = _addmm_nvfp4_dispatch(grad_t_nvfp4, input_t_nvfp4.t(), None)

        # Bias gradient is just sum of grad_output along batch dims
        grad_bias = (
            grad_output_hp.sum(dim=tuple(range(grad_output_hp.dim() - 1)))
            if ctx.has_bias
            else None
        )

        return grad_input, grad_weight, grad_bias, None
