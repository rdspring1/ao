# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rowwise-cast + columnwise-RHT quantize, dynamic signs. §11.9 (§2).

The V2 forward activation quantize, producing per group both operands the recipe
saves from one read of ``A``:

    row_fp4       = nvfp4_cast(A_g)             # rowwise 1x16, no transform
    col_fp4_rht_t = nvfp4_cast(A_g.t() @ R_m)   # columnwise transposed, RHT-128

The RHT-128 dynamic-sign twin of the shipped ``triton_group_rht_quantize_row_col``;
see ``group_row_cast_col_rht_amax_triton`` for why V2 needs its own op rather than
an extra argument on the shipped one.

Both operands are RTNE. Design doc §3's stochastic-rounding variant is not implemented
here: V2 casts only the forward activation through this op, and its backward gradient
goes through MS-EDEN (§11.3) instead, so nothing in the recipe would set the flag.

A dense linear is the degenerate ``num_tensors = 1`` case: ``offsets = [M]``.
"""

from typing import Optional

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

from .group_hadamard_utils import _validate_graph_amax

RHT_SIZE = 128

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (  # noqa: F401
        BLOCK_M,
        BLOCK_N,
        _get_group_idx_binary,
        _validate_grouped_hadamard_inputs,
    )
    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (  # noqa: F401
        _nvfp4_quantize,
        _pack_fp4,
        _store_grouped_scales_swizzle,
        _store_scales_swizzle,
        _swizzle_scales,
        get_dynamic_rht_matrix,
    )

    # num_warps pinned at 4: the register-heavy quantize body over-subscribes at 8,
    # and the 8-warp win is M-dependent. Since M is dropped from the autotune key a
    # single config is cached across all M, so an 8-warp win at a small first-seen M
    # would silently poison large-M steps. Mirrors _GROUP_QUANTIZE_CONFIGS.
    _GROUP_ROW_CAST_COL_RHT_QUANTIZE_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 3, 4)
    ]

    @triton.autotune(
        configs=_GROUP_ROW_CAST_COL_RHT_QUANTIZE_CONFIGS,
        key=["N", "FAST_MATH"],
    )
    @triton.jit
    def _group_row_cast_col_rht_quantize_kernel(
        a_ptr,
        b_ptr,
        qa_ptr,
        sfa_ptr,
        offsets_ptr,
        global_amax_row_ptr,
        global_amax_col_ptr,
        qa_t_ptr,
        sfa_t_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        FAST_MATH: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped fused RHT-128 columnwise + plain rowwise NVFP4 quantization.

        Flat grid ``(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)``. Both outputs come from
        one load of ``a``, which is the point of fusing them. Sketch:

            token_tile_idx = (tile_idx // num_tiles_hidden).to(tl.int64)   # int64!
            token_offset = token_tile_idx * BLOCK_M
            if token_offset < tl.load(logical_packed_length_ptr):
                group_idx = _get_group_idx_binary(token_offset, offsets_ptr, num_tensors)
                a = <load (BLOCK_M, BLOCK_N) tile>
                hadamard = <load (RHT_SIZE, RHT_SIZE) from b_ptr>

                # --- columnwise (wgrad operand) ---
                a_t_r = tl.reshape(tl.trans(a), [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
                a_t_rht = tl.dot(a_t_r, hadamard)
                if not FAST_MATH:          # TE fast math consumes the fp32 accumulator
                    a_t_rht = a_t_rht.to(tl.bfloat16)
                col_sf, col_scaled = _nvfp4_quantize(
                    a_t_rht, tl.load(global_amax_col_ptr + group_idx),
                    BLOCK_N, BLOCK_M, FAST_MATH)
                col_fp4 = _pack_fp4(col_scaled, BLOCK_N, BLOCK_M, False, 0, 0, tile_idx)
                _store_grouped_scales_swizzle(...)   # token axis is inner -> per-group
                <store col_fp4 at qa_t_ptr[n, m], stride M // 2>

                # --- rowwise (forward operand) ---
                row_sf, row_scaled = _nvfp4_quantize(
                    a, tl.load(global_amax_row_ptr + group_idx),
                    BLOCK_M, BLOCK_N, FAST_MATH)
                row_fp4 = _pack_fp4(row_scaled, BLOCK_M, BLOCK_N, False, 0, 0, tile_idx)
                _store_scales_swizzle(...)           # token axis is outer -> contiguous
                <store row_fp4 at qa_ptr[m, n], stride N // 2>

        The two scale stores differ on purpose. Columnwise puts the grouped token axis
        on the inner (64-blocked) side, so the swizzle tiling restarts at every group
        boundary; rowwise has it on the outer axis, where a group is already contiguous.

        Rows at or beyond ``logical_packed_length`` are storage only and must be left
        zero-filled, not quantized.
        """
        # TODO(nvfp4-v2): implement. See design doc §11.9 / §2.
        tl.static_assert(
            False, "_group_row_cast_col_rht_quantize_kernel is not implemented"
        )

    @torch.library.custom_op(
        "torchao::triton_group_row_cast_col_rht_quantize", mutates_args=()
    )
    def triton_group_row_cast_col_rht_quantize(
        A: torch.Tensor,
        sign_vector: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax: torch.Tensor,
        d_global_amax: torch.Tensor,
        logical_packed_length: Optional[torch.Tensor] = None,
        use_fast_math: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Grouped fused RHT-128 columnwise + plain rowwise NVFP4 quantize.

        Args:
            A: packed ``(sum_M, N)`` bfloat16, row-major.
            sign_vector: ``(128,)`` {-1, +1} device tensor, updated in place by the
                cadence manager.
            offsets: int32 cumulative row-end offsets, one per group.
            num_tensors: number of groups.
            packed_sequence_length: allocated row capacity of A.
            hidden_size: number of columns in A.
            shape_rep: SAME_BOTH_DIMS or VARYING_FIRST_DIM.
            a_global_amax: ``(num_tensors,)`` float32 rowwise amaxes -- the second
                element of the ``triton_group_row_cast_col_rht_amax`` return.
            d_global_amax: ``(num_tensors,)`` float32 columnwise post-RHT amaxes --
                the first element of that return. Taken post-RHT on purpose: a
                pre-RHT amax under-bounds the rotated tile and every block scale
                saturates.
            logical_packed_length: one-element int32 CUDA tensor, ``offsets[-1]``.
            use_fast_math: consume the FP32 RHT accumulator directly and take an
                approximate reciprocal, matching TE under ``NVTE_USE_FAST_MATH=1``.

        Returns:
            ``(row_fp4, row_sf, col_fp4_rht_t, col_sf_rht_t)`` -- **rowwise first**,
            matching the shipped ``triton_group_rht_quantize_row_col``:
              - ``(sum_M, N//2)`` uint8 rowwise codes.
              - ``(sum_M, N//16)`` float8_e4m3fn rowwise scales (swizzled bytes,
                returned under their logical 2D view).
              - ``(N, sum_M//2)`` uint8 columnwise transposed codes.
              - ``(N, sum_M//16)`` float8_e4m3fn columnwise scales.
        """
        if sign_vector.ndim != 1 or sign_vector.numel() != RHT_SIZE:
            raise ValueError(
                f"sign_vector must be a ({RHT_SIZE},) tensor, "
                f"got shape {tuple(sign_vector.shape)}"
            )
        if not sign_vector.is_cuda or sign_vector.device != A.device:
            raise ValueError("sign_vector must be on the same device as A")
        B = get_dynamic_rht_matrix(sign_vector, torch.bfloat16)
        _validate_grouped_hadamard_inputs(
            A,
            B,
            offsets,
            num_tensors,
            packed_sequence_length,
            hidden_size,
            shape_rep,
            logical_packed_length,
            rht_size=RHT_SIZE,
        )

        qa_base = torch.empty(
            (packed_sequence_length, hidden_size // 2),
            dtype=torch.uint8,
            device=A.device,
        )
        row_amax = _validate_graph_amax(
            a_global_amax, "a_global_amax", num_tensors, A.device
        )
        sfa_storage = torch.empty(
            (packed_sequence_length // 128, hidden_size // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )
        sfa_return = sfa_storage.view(packed_sequence_length, hidden_size // 16)

        qd = torch.empty(
            (hidden_size, packed_sequence_length // 2),
            dtype=torch.uint8,
            device=A.device,
        )
        col_amax = _validate_graph_amax(
            d_global_amax, "d_global_amax", num_tensors, A.device
        )
        sfd_storage = torch.empty(
            (hidden_size // 128, packed_sequence_length // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=A.device,
        )
        sfd_return = sfd_storage.view(hidden_size, packed_sequence_length // 16)

        m, n = A.shape
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_cast_col_rht_quantize_kernel[grid](
            A,
            B,
            qa_base,
            sfa_storage,
            offsets,
            row_amax,
            col_amax,
            qd,
            sfd_storage,
            m,
            n,
            num_tensors=num_tensors,
            FAST_MATH=use_fast_math,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
            logical_packed_length_ptr=logical_packed_length,
        )
        return qa_base, sfa_return, qd, sfd_return

    @triton_group_row_cast_col_rht_quantize.register_fake
    def _(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        a_global_amax,
        d_global_amax,
        logical_packed_length=None,
        use_fast_math=False,
    ):
        qa_base = A.new_empty(
            (packed_sequence_length, hidden_size // 2), dtype=torch.uint8
        )
        sfa = A.new_empty(
            (packed_sequence_length, hidden_size // 16), dtype=torch.float8_e4m3fn
        )
        qd = A.new_empty((hidden_size, packed_sequence_length // 2), dtype=torch.uint8)
        sfd = A.new_empty(
            (hidden_size, packed_sequence_length // 16), dtype=torch.float8_e4m3fn
        )
        return qa_base, sfa, qd, sfd

else:

    def triton_group_row_cast_col_rht_quantize(
        A: torch.Tensor,
        sign_vector: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        a_global_amax: torch.Tensor,
        d_global_amax: torch.Tensor,
        logical_packed_length: Optional[torch.Tensor] = None,
        use_fast_math: bool = False,
    ):
        raise NotImplementedError(
            "triton_group_row_cast_col_rht_quantize requires torch 2.10.0+ and "
            "triton installed"
        )
