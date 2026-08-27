# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped lazy columnwise NVFP4 weight requantization.

The V1_REQUANT backward weight path. Forward saved only the packed rowwise weight
(``group_row_cast_quantize``); these two ops rebuild the transposed operand
the dgrad GEMM needs, per expert:

    # on chip, per tile
    W_qdq        = dequantize(row_fp4_w, row_sf_w, global_amax)
    amax_w_qdq_t = amax(abs(W_qdq.bf16().t()))
    col_fp4_w_t  = nvfp4_cast(W_qdq.bf16().t())

The original BF16 weight is **never** re-read. Deriving the backward operand from
the quantized forward weight is what makes the forward and dgrad GEMMs agree on
one and the same ``W_qdq`` -- the invariant that V1's 2D 16x16 quantize got from a
shared scale byte instead.

Both ops live in one file because they share ``_load_requant_weight_tile``. That
sharing is a correctness requirement, not an optimization: if the amax pass and the
quantize pass reconstructed ``W_qdq`` differently, ``amax_w_qdq_t`` would not bound
the tensor actually being quantized and both gradients would come out biased low.

No sign vector: these apply no transform. The rotated twins are in
``group_col_rht_requantize_triton.py``.

A dense linear is the degenerate ``num_experts = 1`` case.
"""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

from .group_hadamard_utils import (
    _validate_requant_amax,
    _validate_requant_weight_inputs,
)

BLOCK_M = 128
BLOCK_N = 128

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (  # noqa: F401
        _load_scales_swizzle,
        _nvfp4_global_scales,
        _nvfp4_quantize,
        _rescale_fp4,
        _store_scales_swizzle,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
    )

    # Weights are static, so M is a safe autotune key here (unlike the activation
    # kernels, whose token count varies every step).
    _GROUP_COL_CAST_REQUANT_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.jit
    def _load_requant_weight_tile(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        expert,
        pid_m,
        pid_n,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Reconstruct one ``W_qdq`` tile on chip and return it transposed, in bf16.

        Shared by both kernels below -- see the module docstring for why that is
        load-bearing. Returns a ``(BLOCK_N, BLOCK_M)`` bf16 tile.

        The bf16 round-through before the transpose is required, not incidental:
        the reference takes it, and dropping it makes the codes differ.
        """

        # Load packed fp4 codes
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        packed_offsets = offs_m[:, None] * (N // 2) + packed_inner[None, :]
        qw_expert_ptr = qw_ptr + expert * M * (N // 2)
        qw = tl.load(qw_expert_ptr + packed_offsets)

        # Load swizzled scales
        sfw_expert_stride = (M // 128) * (N // 64) * 32 * 16
        sfw_expert_ptr = sfw_ptr + expert * sfw_expert_stride
        sfw = _load_scales_swizzle(
            sfw_expert_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )

        # Load global amax precomputed in forward pass. Per expert -- a reduction
        # across experts would decode every expert with expert 0's scale.
        FP8_E4M3_MAX: tl.constexpr = 448.0
        amax_w = tl.load(global_amax_ptr + expert)
        _, global_decode_scale = _nvfp4_global_scales(amax_w, FP8_E4M3_MAX)
        dequant_w = _rescale_fp4(qw, sfw, global_decode_scale, BLOCK_M, BLOCK_N)

        # A NaN or inf global_amax must reconstruct to zero, not propagate.
        valid_amax = (amax_w == amax_w) & (tl.abs(amax_w) != float("inf"))
        dequant_w = tl.where(valid_amax, dequant_w, 0.0).to(tl.bfloat16)
        return tl.trans(tl.reshape(dequant_w, [BLOCK_M, BLOCK_N]))

    @triton.autotune(configs=_GROUP_COL_CAST_REQUANT_CONFIGS, key=["M", "N"])
    @triton.jit
    def _group_col_cast_requant_amax_kernel(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        amax_dw_t_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Per-expert amax over the reconstructed weight transpose.

        Grid ``(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), E)``; reduce into slot ``expert``
        with ``tl.atomic_max``. Re-inject NaN explicitly -- ``tl.max`` drops it.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2).to(tl.int64)
        dw_t = _load_requant_weight_tile(
            qw_ptr,
            sfw_ptr,
            global_amax_ptr,
            expert,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )
        amax_dw_t = tl.max(tl.abs(dw_t))

        # Re-inject NaN explicitly -- ``tl.max`` drops it. The probe has to run on the
        # tile: by this point ``amax_dw_t`` is the reduction that dropped the NaN.
        tile_has_nan = tl.max((dw_t != dw_t).to(tl.int32))
        amax_dw_t = tl.where(tile_has_nan != 0, float("nan"), amax_dw_t)

        tl.atomic_max(amax_dw_t_ptr + expert, amax_dw_t.to(tl.float32))

    @triton.autotune(configs=_GROUP_COL_CAST_REQUANT_CONFIGS, key=["M", "N"])
    @triton.jit
    def _group_col_cast_requantize_kernel(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        amax_dw_t_ptr,
        qdq_w_t_ptr,
        sfw_t_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Per-expert rowwise 1x16 NVFP4 quantization of the reconstructed transpose
        Same reconstruction as the amax kernel, then quantize and store transposed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2).to(tl.int64)

        dw_t = _load_requant_weight_tile(
            qw_ptr,
            sfw_ptr,
            global_amax_ptr,
            expert,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )

        global_amax = tl.load(amax_dw_t_ptr + expert)
        sfw_t, qdq_w_t = _nvfp4_quantize(dw_t, global_amax, BLOCK_N, BLOCK_M)

        # Pack FP4 values into uint8 -- non-transposed: (BLOCK_M, BLOCK_N//2, 2).
        qdq_w_t_pairs = qdq_w_t.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        qdq_w_t_fp4x2 = convert_8xfp32_to_4xfp4_packed(qdq_w_t_pairs)

        # Store packed FP4 values in this expert's rowwise output.
        outer = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_inner = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
        packed_offsets = outer[:, None] * (M // 2) + packed_inner[None, :]

        # Shift base pointers for packed NVFP4 tensors to this expert.
        qdq_w_t_expert_ptr = qdq_w_t_ptr + expert * N * (M // 2)
        tl.store(qdq_w_t_expert_ptr + packed_offsets, qdq_w_t_fp4x2)

        # Shift base pointers for FP8 scale factors to this expert.
        sfw_t_expert_stride = (N // 128) * (M // 64) * 32 * 16
        sfw_t_expert_ptr = sfw_t_ptr + expert * sfw_t_expert_stride

        swizzle_sfw_t = _swizzle_scales(sfw_t, BLOCK_N, BLOCK_M)
        _store_scales_swizzle(
            swizzle_sfw_t,
            sfw_t_expert_ptr,
            pid_n,
            pid_m,
            N,
            M,
            BLOCK_N,
            BLOCK_M,
        )

    @torch.library.custom_op(
        "torchao::triton_group_col_cast_requant_amax", mutates_args=()
    )
    def triton_group_col_cast_requant_amax(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> torch.Tensor:
        """Per-expert amax of the dequantized forward weight, transposed. §11.6.

        Args:
            row_fp4_w: ``(E, M, N//2)`` uint8 packed FP4 codes from §11.1.
            row_sf_w: ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.
            global_amax: ``(E,)`` float32 amax of the *original* weight, the one §11.1
                encoded with.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            ``(E,)`` float32 where ``out[g] = dequantize(row_fp4_w[g]).bf16().abs().amax()``.

        This is not ``global_amax``, and reusing ``global_amax`` in its place is a bug
        this op exists to avoid: ``global_amax`` bounds the original weight, while the
        tensor about to be quantized is the quantized-dequantized one. The two are
        equal only when the weight was already exactly representable in NVFP4.

        Takes no sign vector: it applies no transform. Because a transpose does not
        change the set of elements this is mathematically ``amax(abs(W_qdq))``; it is
        spelled as the transpose to stay the exact twin of §11.4.
        """
        E, M, N = _validate_requant_weight_inputs(
            row_fp4_w,
            row_sf_w,
            global_amax,
            num_tensors,
            "triton_group_col_cast_requant_amax",
        )
        # Zero-initialized: the kernel reduces with atomic_max, so repeated launches
        # are idempotent and autotuning needs no reset_to_zero.
        amax_w_qdq_t = torch.zeros((E,), dtype=torch.float32, device=row_fp4_w.device)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_col_cast_requant_amax_kernel[grid](
            row_fp4_w,
            row_sf_w,
            global_amax,
            amax_w_qdq_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return amax_w_qdq_t

    @triton_group_col_cast_requant_amax.register_fake
    def _(row_fp4_w, row_sf_w, global_amax, num_tensors):
        return row_fp4_w.new_empty((row_fp4_w.shape[0],), dtype=torch.float32)

    @torch.library.custom_op(
        "torchao::triton_group_col_cast_requantize", mutates_args=()
    )
    def triton_group_col_cast_requantize(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        amax_w_qdq_t: torch.Tensor,
        num_tensors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-expert columnwise NVFP4 requantization of the forward weight. §11.7.

        Args:
            row_fp4_w: ``(E, M, N//2)`` uint8 packed FP4 codes from §11.1.
            row_sf_w: ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.
            global_amax: ``(E,)`` float32 amax of the original weight.
            amax_w_qdq_t: ``(E,)`` float32 from §11.6. Must come from §11.6 and not be
                guessed: too small and every block scale saturates.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            A 2-tuple containing:
              - ``(E, N, M//2)`` uint8 columnwise FP4 codes (rowwise ``W_qdq.T``).
              - ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn swizzled scales.

        Consumed immediately by dgrad and not retained. Unlike the 2D scheme this
        replaces, the rowwise and columnwise scale bytes now differ -- but both decode
        to one ``W_qdq``, which is the stronger property.
        """
        E, M, N = _validate_requant_weight_inputs(
            row_fp4_w,
            row_sf_w,
            global_amax,
            num_tensors,
            "triton_group_col_cast_requantize",
        )
        _validate_requant_amax(amax_w_qdq_t, "amax_w_qdq_t", E, row_fp4_w.device)

        qa_t = torch.empty((E, N, M // 2), dtype=torch.uint8, device=row_fp4_w.device)
        sfa_t = torch.empty(
            (E, N // 128, M // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=row_fp4_w.device,
        )
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_col_cast_requantize_kernel[grid](
            row_fp4_w,
            row_sf_w,
            global_amax,
            amax_w_qdq_t,
            qa_t,
            sfa_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return qa_t, sfa_t

    @triton_group_col_cast_requantize.register_fake
    def _(row_fp4_w, row_sf_w, global_amax, amax_w_qdq_t, num_tensors):
        E, M, packed_N = row_fp4_w.shape
        N = packed_N * 2
        qa_t = row_fp4_w.new_empty((E, N, M // 2), dtype=torch.uint8)
        sfa_t = row_fp4_w.new_empty(
            (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn
        )
        return qa_t, sfa_t

else:

    def triton_group_col_cast_requant_amax(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_group_col_cast_requant_amax requires torch 2.10.0+ and Triton"
        )

    def triton_group_col_cast_requantize(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        amax_w_qdq_t: torch.Tensor,
        num_tensors: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_col_cast_requantize requires torch 2.10.0+ and Triton"
        )
