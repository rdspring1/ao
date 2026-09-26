# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped 1D (1x16) rowwise NVFP4 weight quantization.

Replaces ``triton_group_weight_quantize_2d`` in the forward of the V1_REQUANT and
V2 recipes. Two differences from the 2D kernel it replaces:

* **1x16 blocks, not 16x16.** The 2D kernel computes one scale per 16x16 tile and
  broadcasts it to all 16 rows, so ``W`` and ``W.T`` are forced to share a scale
  byte. Here every row of 16 gets its own scale, so ``N * D / 16`` logical scales
  instead of ``N * D / 256``.
* **Rowwise only.** No columnwise output is produced. The transposed operand the
  dgrad GEMM needs is rebuilt in backward from these packed codes, by
  ``group_col_cast_requantize`` or ``group_col_rht_requantize``.
  Both GEMMs then decode to one and the same ``W_qdq``, which is a stronger
  property than the shared scale byte it replaces.

Weights never use stochastic rounding: RTNE only, so there is no ``rng_state``.

A dense linear is the degenerate ``num_experts = 1`` case -- pass ``w.unsqueeze(0)``.
"""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

BLOCK_M = 128
BLOCK_N = 128

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (  # noqa: F401
        _nvfp4_quantize,
        _store_scales_swizzle,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
    )
    from torchao.utils import is_sm_at_least_100

    # Autotune num_warps/num_stages only; BLOCK is held at 128 by the swizzled
    # scale layout. Mirrors _GROUP_QUANTIZE_2D_CONFIGS -- weights are static, so
    # unlike the activation kernels M is a safe autotune key here.
    _GROUP_ROW_CAST_QUANTIZE_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.autotune(configs=_GROUP_ROW_CAST_QUANTIZE_CONFIGS, key=["M", "N"])
    @triton.jit
    def _group_row_cast_quantize_kernel(
        a_ptr,
        global_amax_ptr,
        qa_ptr,
        sfa_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Per-expert rowwise 1x16 NVFP4 quantization -- one tile per CTA.

        Grid is ``(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), E)``; ``program_id(2)`` is
        the expert. Expert base offsets must be widened with ``.to(tl.int64)``
        before multiplying by a row stride: at DeepSeek expert shapes ``E * M * N``
        passes 2**31 and a 32-bit product wraps to a bad address.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2).to(tl.int64)

        # Get global amax for this expert (float32).
        global_amax = tl.load(global_amax_ptr + expert)

        # Shift base pointers for packed NVFP4 tensors to this expert.
        qa_expert_ptr = qa_ptr + expert * M * (N // 2)

        # Shift base pointers for FP8 scale factors to this expert.
        sfa_expert_stride = (M // 128) * (N // 64) * 32 * 16
        sfa_expert_ptr = sfa_ptr + expert * sfa_expert_stride

        # Load a 2D (BLOCK_M, BLOCK_N) tile for this expert.
        a_expert_ptr = a_ptr + expert * M * N
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        a = tl.load(a_expert_ptr + offs_m[:, None] * N + offs_n[None, :])

        # Compute per-1x16-vector scales and scaled values.
        sfa, qa = _nvfp4_quantize(a, global_amax, BLOCK_M, BLOCK_N)

        # Pack FP4 values into uint8 -- non-transposed: (BLOCK_M, BLOCK_N//2, 2).
        qa_pairs = qa.reshape(BLOCK_M, BLOCK_N // 2, 2).split()
        qa_fp4x2 = convert_8xfp32_to_4xfp4_packed(qa_pairs)

        # Store packed FP4 values in this expert's rowwise output.
        outer = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        packed_offsets = outer[:, None] * (N // 2) + packed_inner[None, :]
        tl.store(qa_expert_ptr + packed_offsets, qa_fp4x2)

        swizzle_sfa = _swizzle_scales(sfa, BLOCK_M, BLOCK_N)
        _store_scales_swizzle(
            swizzle_sfa,
            sfa_expert_ptr,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
        )

    @torch.library.custom_op("torchao::triton_group_row_cast_quantize", mutates_args=())
    def triton_group_row_cast_quantize(
        A: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-expert rowwise 1x16 NVFP4 E2M1 weight quantization. RTNE, no RHT.

        Args:
            A: Dense ``(E, M, N)`` BF16 weights, contiguous. M and N must be
                divisible by 128.
            global_amax: ``(E,)`` float32 per-expert absolute maxima, as produced by
                ``triton_group_weight_amax``. Expert ``g`` is quantized with
                ``global_amax[g]``, never a reduction across experts.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            A 2-tuple containing:
              - ``(E, M, N//2)`` uint8 rowwise FP4 codes.
              - ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.

            The arity is two, not four: there is deliberately no columnwise output.
        """
        if not is_sm_at_least_100():
            raise NotImplementedError("triton_group_row_cast_quantize requires SM100+")
        if A.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16, got {A.dtype}")
        if A.ndim != 3:
            raise ValueError("Tensor A must be 3-D")
        if not A.is_contiguous():
            raise ValueError("A must be contiguous")

        E, M, N = A.shape
        if E != num_tensors:
            raise ValueError(f"Expected {num_tensors} experts, got {E}")
        if global_amax.shape != (E,):
            raise ValueError(f"global_amax must have shape ({E},)")
        if global_amax.dtype != torch.float32:
            raise ValueError(f"Expected float32 global_amax, got {global_amax.dtype}")
        if not global_amax.is_cuda or global_amax.device != A.device:
            raise ValueError("global_amax must be on the same device as A")
        if not global_amax.is_contiguous():
            raise ValueError("global_amax must be contiguous")
        if M % BLOCK_M != 0 or N % BLOCK_N != 0:
            raise ValueError(
                f"Expected M divisible by {BLOCK_M} and N divisible by {BLOCK_N}, "
                f"got M={M}, N={N}"
            )

        qa = torch.empty((E, M, N // 2), dtype=torch.uint8, device=A.device)
        sfa = torch.empty(
            (E, M // 128, N // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_row_cast_quantize_kernel[grid](
            A,
            global_amax,
            qa,
            sfa,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        return qa, sfa

    @triton_group_row_cast_quantize.register_fake
    def _(A, global_amax, num_tensors):
        E, M, N = A.shape
        qa = A.new_empty((E, M, N // 2), dtype=torch.uint8)
        sfa = A.new_empty((E, M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
        return qa, sfa

else:

    def triton_group_row_cast_quantize(
        A: torch.Tensor,
        global_amax: torch.Tensor,
        num_tensors: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_row_cast_quantize requires torch 2.10.0+ and Triton"
        )
