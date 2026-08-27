# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rowwise-RHT + columnwise-RHT amax. Design doc §11.2 (§5).

The V2 backward gradient amax pass. Unlike §11.8, **both** axes are transformed,
with **independent** sign vectors:

    R_n              = diag(dgrad_rht) @ H_128 / sqrt(128)
    R_m              = diag(wgrad_rht) @ H_128 / sqrt(128)
    amax_rht_dy[g]   = amax(abs(dy_g @ R_n))
    amax_rht_dy_t[g] = amax(abs(dy_g.t() @ R_m))

The two sign vectors must not be swapped: ``dgrad_rht`` rotates the row axis for the
dgrad operand and ``wgrad_rht`` rotates the transposed axis for the wgrad operand.
Each has to match the sign vector used on the *other* operand of its GEMM for the
transform to cancel, so a crossed pair produces a wrong gradient rather than an
error. The sign vectors are shared across groups; only the data is grouped.

Both are live device buffers, resampled on different cadences (``dgrad_rht`` per
optimizer step, ``wgrad_rht`` per accumulation microbatch), so the RHT matrices are
formed per launch from the fixed cached H128 -- never memoized by sign value.

A dense linear is the degenerate ``num_tensors = 1`` case: ``offsets = [M]``.
"""

from typing import Optional

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

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
        get_dynamic_rht_matrix,
    )

    # M excluded from the key: variable token counts would re-benchmark every step
    # and break CUDA-graph capture on a cold key, for no per-tile config gain.
    _GROUP_ROW_RHT_COL_RHT_AMAX_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.autotune(configs=_GROUP_ROW_RHT_COL_RHT_AMAX_CONFIGS, key=["N"])
    @triton.jit
    def _group_row_rht_col_rht_amax_kernel(
        a_ptr,
        row_rht_ptr,
        col_rht_ptr,
        offsets_ptr,
        amax_row_rht_ptr,
        amax_col_rht_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped RHT-128 amax on both axes, one tile per CTA.

        Two independent Hadamard operands are loaded, not one. Sketch:

            token_tile_idx = (tile_idx // num_tiles_hidden).to(tl.int64)   # int64!
            token_offset = token_tile_idx * BLOCK_M
            if token_offset < tl.load(logical_packed_length_ptr):
                group_idx = _get_group_idx_binary(token_offset, offsets_ptr, num_tensors)
                a = <load (BLOCK_M, BLOCK_N) tile>
                row_rht = <load (RHT_SIZE, RHT_SIZE) from row_rht_ptr>   # R_n
                col_rht = <load (RHT_SIZE, RHT_SIZE) from col_rht_ptr>   # R_m

                a_r = tl.reshape(a, [BLOCK_M * BLOCK_N // RHT_SIZE, RHT_SIZE])
                a_rht = tl.dot(a_r, row_rht).to(tl.bfloat16)
                <atomic_max abs(a_rht) into amax_row_rht_ptr + group_idx>

                a_t_r = tl.reshape(tl.trans(a), [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
                a_t_rht = tl.dot(a_t_r, col_rht).to(tl.bfloat16)
                <atomic_max abs(a_t_rht) into amax_col_rht_ptr + group_idx>

        ``row_rht`` multiplies the un-transposed tile and ``col_rht`` the transposed
        one. Wiring them the other way round is the single most likely bug here and
        produces no error, only a wrong gradient.

        Both reductions must re-inject NaN explicitly, since ``tl.max`` drops it.
        """
        # TODO(nvfp4-v2): implement. See design doc §11.2 / §5.
        tl.static_assert(False, "_group_row_rht_col_rht_amax_kernel is not implemented")

    @torch.library.custom_op(
        "torchao::triton_group_row_rht_col_rht_amax", mutates_args=()
    )
    def triton_group_row_rht_col_rht_amax(
        dy: torch.Tensor,
        dgrad_rht: torch.Tensor,
        wgrad_rht: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group global amaxes for the V2 backward MS-EDEN operands.

        Args:
            dy: packed ``(sum_M, N)`` bfloat16 gradient, row-major.
            dgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the **row** axis.
            wgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the **transposed**
                axis. Independent of ``dgrad_rht``; equality between them is never
                assumed anywhere.
            offsets: int32 cumulative row-end offsets, one per group.
            num_tensors: number of groups.
            packed_sequence_length: allocated row capacity of dy.
            hidden_size: number of columns in dy.
            shape_rep: SAME_BOTH_DIMS or VARYING_FIRST_DIM.
            logical_packed_length: one-element int32 CUDA tensor, ``offsets[-1]``.

        Returns:
            ``(amax_rht_dy, amax_rht_dy_t)``, each ``(num_tensors,)`` float32 --
            **rowwise first**, following the op's name:
              - ``amax_rht_dy[g]   = amax(abs(dy_g @ R_n))``     (dgrad operand)
              - ``amax_rht_dy_t[g] = amax(abs(dy_g.t() @ R_m))`` (wgrad operand)

            Note this differs from ``triton_group_rht_amax``, which returns the
            transformed columnwise amax first because its rowwise value is an
            untransformed plain amax. Here both are transformed.
        """
        for name, sv in (("dgrad_rht", dgrad_rht), ("wgrad_rht", wgrad_rht)):
            if sv.ndim != 1 or sv.numel() != RHT_SIZE:
                raise ValueError(
                    f"{name} must be a ({RHT_SIZE},) tensor, got shape {tuple(sv.shape)}"
                )
            if not sv.is_cuda or sv.device != dy.device:
                raise ValueError(f"{name} must be on the same device as dy")
        row_rht = get_dynamic_rht_matrix(dgrad_rht, torch.bfloat16)
        col_rht = get_dynamic_rht_matrix(wgrad_rht, torch.bfloat16)
        _validate_grouped_hadamard_inputs(
            dy,
            row_rht,
            offsets,
            num_tensors,
            packed_sequence_length,
            hidden_size,
            shape_rep,
            logical_packed_length,
            rht_size=RHT_SIZE,
        )

        amax_rht_dy = torch.zeros((num_tensors,), dtype=torch.float32, device=dy.device)
        amax_rht_dy_t = torch.zeros(
            (num_tensors,), dtype=torch.float32, device=dy.device
        )

        m, n = dy.shape
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_rht_col_rht_amax_kernel[grid](
            dy,
            row_rht,
            col_rht,
            offsets,
            amax_rht_dy,
            amax_rht_dy_t,
            m,
            n,
            num_tensors=num_tensors,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
            logical_packed_length_ptr=logical_packed_length,
        )
        return amax_rht_dy, amax_rht_dy_t

    @triton_group_row_rht_col_rht_amax.register_fake
    def _(
        dy,
        dgrad_rht,
        wgrad_rht,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length=None,
    ):
        amax_rht_dy = dy.new_empty((num_tensors,), dtype=torch.float32)
        amax_rht_dy_t = dy.new_empty((num_tensors,), dtype=torch.float32)
        return amax_rht_dy, amax_rht_dy_t

else:

    def triton_group_row_rht_col_rht_amax(
        dy: torch.Tensor,
        dgrad_rht: torch.Tensor,
        wgrad_rht: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_row_rht_col_rht_amax requires torch 2.10.0+ and triton "
            "installed"
        )
