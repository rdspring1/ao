# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped lazy columnwise RHT-128 weight requantization.

The V2 backward weight path with the Random Hadamard Transform added:

    W_qdq            = dequantize(row_fp4_w, row_sf_w, global_amax)   # on chip
    R_n              = diag(dgrad_rht) @ H_128 / sqrt(128)
    amax_rht_w_qdq_t = amax(abs(W_qdq.bf16().t() @ R_n))              # §11.4
    col_fp4_rht_w_t  = nvfp4_cast(W_qdq.bf16().t() @ R_n)             # §11.5

The rotation is what lets V2's dgrad cancel: the ``dy`` operand carries ``R_n`` too,
so ``(dy @ R_n) @ (w.t() @ R_n).t() = dy @ w``. The sign vector must be the same
``dgrad_rht`` the gradient was rotated with in the same backward pass; a mismatch
produces a wrong ``dx`` and no error.

One ``dgrad_rht`` is shared across all experts, so a single ``B`` operand serves the
whole weight stack. That is the only grouped-plus-RHT combination supported here:
a per-expert ``B`` would need a different operand per group and is not implemented.

Both ops share ``_load_rht_requant_weight_tile`` and that sharing is a correctness
requirement. If the amax pass and the quantize pass reconstructed or rotated
``W_qdq`` differently, the amax would not bound the tensor being quantized. The
reconstruction under it is ``_reconstruct_qdq_weight_tile`` in ``hadamard_utils``,
shared with the un-rotated twins so the invariant holds across the two files too.

Note the decode numerator here is 2688, not 1536: the weight is a *cast* operand
even in V2. Only the MS-EDEN gradient operands use 1536.

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
RHT_SIZE = 128

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (  # noqa: F401
        _nvfp4_quantize,
        _reconstruct_qdq_weight_tile,
        _store_scales_swizzle,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
        get_dynamic_rht_matrix,
    )

    # Weights are static, so M is a safe autotune key here.
    _GROUP_COL_RHT_REQUANT_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.jit
    def _load_rht_requant_weight_tile(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        rht_ptr,
        expert,
        pid_m,
        pid_n,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
    ):
        """Reconstruct one ``W_qdq`` tile on chip, transpose it, and rotate by ``R_n``.

        Shared by both kernels below; see the module docstring for why that matters.
        Returns ``(BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE)`` bf16.

        The reconstruction is ``_reconstruct_qdq_weight_tile``, the same one the cast
        twins use, which is what keeps the two files decoding the saved weight
        identically. Its bf16 round-through *before* the rotation is required to match
        the reference; so is the one after the ``tl.dot``.
        """
        # The reshape below is only correct while an RHT chunk stays inside one row
        # of the transposed tile, whose inner extent is BLOCK_M. Violating this
        # applies the transform across the wrong axis with no error, only a wrong
        # gradient.
        tl.static_assert(
            BLOCK_M % RHT_SIZE == 0, "columnwise RHT requires BLOCK_M % RHT_SIZE == 0"
        )

        w_qdq = _reconstruct_qdq_weight_tile(
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

        rht_range = tl.arange(0, RHT_SIZE)
        rht_offsets = rht_range[:, None] * RHT_SIZE + rht_range[None, :]
        rht = tl.load(rht_ptr + rht_offsets)

        w_qdq_t = tl.reshape(
            tl.trans(w_qdq), [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE]
        )
        return tl.dot(w_qdq_t, rht).to(tl.bfloat16)

    @triton.autotune(configs=_GROUP_COL_RHT_REQUANT_CONFIGS, key=["M", "N"])
    @triton.jit
    def _group_col_rht_requant_amax_kernel(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        rht_ptr,
        amax_rht_dw_t_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
    ):
        """Per-expert amax over the rotated, reconstructed weight transpose.

        Grid ``(cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), E)``; ``tl.atomic_max`` into slot
        ``expert``. Re-inject NaN explicitly -- ``tl.max`` drops it. A NaN or inf
        ``global_amax`` must reconstruct to zero and leave the amax at zero, not NaN.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2).to(tl.int64)
        rht_dw_t = _load_rht_requant_weight_tile(
            qw_ptr,
            sfw_ptr,
            global_amax_ptr,
            rht_ptr,
            expert,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            RHT_SIZE,
        )
        amax_rht_dw_t = tl.max(tl.abs(rht_dw_t))

        # Re-inject NaN explicitly -- ``tl.max`` drops it. The probe has to run on
        # the  tile: by this point ``amax_rht_dw_t`` is the reduction that dropped
        # the NaN.
        tile_has_nan = tl.max((rht_dw_t != rht_dw_t).to(tl.int32))
        amax_rht_dw_t = tl.where(tile_has_nan != 0, float("nan"), amax_rht_dw_t)

        tl.atomic_max(amax_rht_dw_t_ptr + expert, amax_rht_dw_t.to(tl.float32))

    @triton.autotune(configs=_GROUP_COL_RHT_REQUANT_CONFIGS, key=["M", "N"])
    @triton.jit
    def _group_col_rht_requantize_kernel(
        qw_ptr,
        sfw_ptr,
        global_amax_ptr,
        rht_ptr,
        amax_rht_dw_t_ptr,
        rht_qdq_w_t_ptr,
        sfw_t_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
    ):
        """Per-expert NVFP4 quantization of the rotated, reconstructed transpose.

        Same reconstruction and rotation as the amax kernel, then quantize and store
        transposed.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.program_id(2).to(tl.int64)

        rht_dw_t = _load_rht_requant_weight_tile(
            qw_ptr,
            sfw_ptr,
            global_amax_ptr,
            rht_ptr,
            expert,
            pid_m,
            pid_n,
            M,
            N,
            BLOCK_M,
            BLOCK_N,
            RHT_SIZE,
        )

        global_amax = tl.load(amax_rht_dw_t_ptr + expert)
        sfw_t, rht_qdq_w_t = _nvfp4_quantize(rht_dw_t, global_amax, BLOCK_N, BLOCK_M)

        # Pack FP4 values into uint8 -- non-transposed: (BLOCK_M, BLOCK_N//2, 2).
        rht_qdq_w_t_pairs = rht_qdq_w_t.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
        rht_qdq_w_t_fp4x2 = convert_8xfp32_to_4xfp4_packed(rht_qdq_w_t_pairs)

        # Store packed FP4 values in this expert's rowwise output.
        outer = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        packed_inner = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
        packed_offsets = outer[:, None] * (M // 2) + packed_inner[None, :]

        # Shift base pointers for packed NVFP4 tensors to this expert.
        rht_qdq_w_t_expert_ptr = rht_qdq_w_t_ptr + expert * N * (M // 2)
        tl.store(rht_qdq_w_t_expert_ptr + packed_offsets, rht_qdq_w_t_fp4x2)

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

    def _rht_operand(dgrad_rht: torch.Tensor, device: torch.device) -> torch.Tensor:
        if dgrad_rht.ndim != 1 or dgrad_rht.numel() != RHT_SIZE:
            raise ValueError(
                f"dgrad_rht must be a ({RHT_SIZE},) tensor, "
                f"got shape {tuple(dgrad_rht.shape)}"
            )
        if not dgrad_rht.is_cuda or dgrad_rht.device != device:
            raise ValueError("dgrad_rht must be on the same device as row_fp4_w")
        return get_dynamic_rht_matrix(dgrad_rht, torch.bfloat16)

    @torch.library.custom_op(
        "torchao::triton_group_col_rht_requant_amax", mutates_args=()
    )
    def triton_group_col_rht_requant_amax(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        dgrad_rht: torch.Tensor,
        num_tensors: int,
    ) -> torch.Tensor:
        """Per-expert amax of the rotated dequantized forward weight transpose. §11.4.

        Args:
            row_fp4_w: ``(E, M, N//2)`` uint8 packed FP4 codes from §11.1.
            row_sf_w: ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.
            global_amax: ``(E,)`` float32 amax of the *original* weight.
            dgrad_rht: ``(128,)`` {-1, +1} device tensor, shared across experts.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            ``(E,)`` float32 where
            ``out[g] = amax(abs(dequantize(row_fp4_w[g]).bf16().t() @ R_n))``.

        Computed from the *quantized* weight, never from the original BF16 one --
        that is what keeps the forward and dgrad GEMMs on one ``W_qdq``. Computing it
        from the BF16 weight gives a different number, which is the discriminating
        test for this op.
        """
        E, M, N = _validate_requant_weight_inputs(
            row_fp4_w,
            row_sf_w,
            global_amax,
            num_tensors,
            "triton_group_col_rht_requant_amax",
        )
        B = _rht_operand(dgrad_rht, row_fp4_w.device)

        # Zero-initialized and reduced by atomic_max: idempotent across autotune runs.
        amax_rht_w_qdq_t = torch.zeros(
            (E,), dtype=torch.float32, device=row_fp4_w.device
        )
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_col_rht_requant_amax_kernel[grid](
            row_fp4_w,
            row_sf_w,
            global_amax,
            B,
            amax_rht_w_qdq_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
        )
        return amax_rht_w_qdq_t

    @triton_group_col_rht_requant_amax.register_fake
    def _(row_fp4_w, row_sf_w, global_amax, dgrad_rht, num_tensors):
        return row_fp4_w.new_empty((row_fp4_w.shape[0],), dtype=torch.float32)

    @torch.library.custom_op(
        "torchao::triton_group_col_rht_requantize", mutates_args=()
    )
    def triton_group_col_rht_requantize(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        amax_rht_w_qdq_t: torch.Tensor,
        dgrad_rht: torch.Tensor,
        num_tensors: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-expert rotated columnwise NVFP4 requantization of the weight. §11.5.

        Args:
            row_fp4_w: ``(E, M, N//2)`` uint8 packed FP4 codes from §11.1.
            row_sf_w: ``(E, M//128, N//64, 32, 16)`` float8_e4m3fn swizzled scales.
            global_amax: ``(E,)`` float32 amax of the original weight.
            amax_rht_w_qdq_t: ``(E,)`` float32 from §11.4. Must come from §11.4 and
                with the same ``dgrad_rht``: halve it and that expert's block scales
                saturate.
            dgrad_rht: ``(128,)`` {-1, +1} device tensor, shared across experts. Must
                be the same vector the gradient was rotated with in this backward.
            num_tensors: Number of experts; must equal ``E``.

        Returns:
            A 2-tuple containing:
              - ``(E, N, M//2)`` uint8 rotated columnwise FP4 codes.
              - ``(E, N//128, M//64, 32, 16)`` float8_e4m3fn swizzled scales.

            Decode with ``NVFP4_CAST_NUMERATOR`` (2688): the weight is a cast operand
            even though the gradient it multiplies is MS-EDEN at 1536.
        """
        E, M, N = _validate_requant_weight_inputs(
            row_fp4_w,
            row_sf_w,
            global_amax,
            num_tensors,
            "triton_group_col_rht_requantize",
        )
        _validate_requant_amax(
            amax_rht_w_qdq_t, "amax_rht_w_qdq_t", E, row_fp4_w.device
        )
        B = _rht_operand(dgrad_rht, row_fp4_w.device)

        qa_t = torch.empty((E, N, M // 2), dtype=torch.uint8, device=row_fp4_w.device)
        sfa_t = torch.empty(
            (E, N // 128, M // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=row_fp4_w.device,
        )
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), E)
        _group_col_rht_requantize_kernel[grid](
            row_fp4_w,
            row_sf_w,
            global_amax,
            B,
            amax_rht_w_qdq_t,
            qa_t,
            sfa_t,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
        )
        return qa_t, sfa_t

    @triton_group_col_rht_requantize.register_fake
    def _(row_fp4_w, row_sf_w, global_amax, amax_rht_w_qdq_t, dgrad_rht, num_tensors):
        E, M, packed_N = row_fp4_w.shape
        N = packed_N * 2
        qa_t = row_fp4_w.new_empty((E, N, M // 2), dtype=torch.uint8)
        sfa_t = row_fp4_w.new_empty(
            (E, N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn
        )
        return qa_t, sfa_t

else:

    def triton_group_col_rht_requant_amax(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        dgrad_rht: torch.Tensor,
        num_tensors: int,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "triton_group_col_rht_requant_amax requires torch 2.10.0+ and Triton"
        )

    def triton_group_col_rht_requantize(
        row_fp4_w: torch.Tensor,
        row_sf_w: torch.Tensor,
        global_amax: torch.Tensor,
        amax_rht_w_qdq_t: torch.Tensor,
        dgrad_rht: torch.Tensor,
        num_tensors: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_col_rht_requantize requires torch 2.10.0+ and Triton"
        )
