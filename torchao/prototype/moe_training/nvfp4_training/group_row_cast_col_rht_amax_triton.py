# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rowwise-cast + columnwise-RHT amax, dynamic signs. Design doc §11.8 (§1).

The V2 forward activation amax pass, computing per group:

    R_m            = diag(sign_vector) @ H_128 / sqrt(128)
    amax_a[g]      = amax(abs(A_g))            # rowwise, no transform
    amax_rht_a_t[g]= amax(abs(A_g.t() @ R_m))  # columnwise transposed, RHT-128

This is the same computation as the shipped ``triton_group_rht_amax``, and differs
from it in exactly two ways, both required by V2:

* **RHT-128, not RHT-16.**
* **The sign vector is a device tensor, not a hashable list.** V2 resamples
  ``wgrad_rht`` every accumulation microbatch, so the RHT matrix cannot be memoized
  by value: ``get_rht_matrix``'s ``lru_cache(maxsize=None)`` would grow one entry per
  resample for the run's lifetime. ``get_dynamic_rht_matrix`` memoizes only the fixed
  H128 and forms ``diag(signs) @ H128`` per call.

It is a separate op rather than optional arguments on the shipped one so both keep a
fixed arity with a simple ``register_fake``, and so recipe V1 stays bit-for-bit
untouched. V1 and V1_REQUANT continue to use ``triton_group_rht_amax`` at RHT-16.

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

    # M (total packed token count) is deliberately excluded from the autotune key:
    # the body is straight-line per 128x128 tile, so M sets only the grid size, not
    # the per-tile num_warps/num_stages optimum. Keying on M would re-benchmark on
    # every step under variable token counts and break CUDA-graph capture on a cold
    # key. Same reasoning as _GROUP_QUANTIZE_CONFIGS in the shipped quantize kernel.
    _GROUP_ROW_CAST_COL_RHT_AMAX_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=nw, num_stages=ns)
        for ns in (2, 3, 4)
        for nw in (4, 8)
    ]

    @triton.autotune(configs=_GROUP_ROW_CAST_COL_RHT_AMAX_CONFIGS, key=["N"])
    @triton.jit
    def _group_row_cast_col_rht_amax_kernel(
        a_ptr,
        b_ptr,
        offsets_ptr,
        global_amax_row_ptr,
        global_amax_col_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped RHT-128 columnwise amax and plain rowwise amax.

        Grid is a flat ``(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)``. Sketch:

            tile_idx = tl.program_id(0)
            num_tiles_hidden = tl.cdiv(N, BLOCK_N)
            # int64: token_tile_idx * BLOCK_M * N overflows int32 past ~2**31.
            token_tile_idx = (tile_idx // num_tiles_hidden).to(tl.int64)
            hidden_tile_idx = tile_idx - token_tile_idx * num_tiles_hidden
            token_offset = token_tile_idx * BLOCK_M
            if token_offset < tl.load(logical_packed_length_ptr):
                group_idx = _get_group_idx_binary(token_offset, offsets_ptr, num_tensors)
                a = <load (BLOCK_M, BLOCK_N) tile>
                hadamard = <load (RHT_SIZE, RHT_SIZE) from b_ptr>
                a_t_r = tl.reshape(tl.trans(a), [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
                a_t_rht = tl.dot(a_t_r, hadamard).to(tl.bfloat16)
                <atomic_max abs(a_t_rht) into global_amax_col_ptr + group_idx>
                <atomic_max abs(a)        into global_amax_row_ptr + group_idx>

        The ``token_offset < logical_packed_length`` guard skips the dispatcher's
        spare allocation capacity: those rows are never initialized and must not
        reach either reduction.

        Both reductions must re-inject NaN explicitly -- ``tl.max`` drops it, so a NaN
        activation would silently produce a finite amax:

            amax = tl.max(tl.max(v, axis=1), axis=0)
            has_nan = tl.max(tl.max((v != v).to(tl.int32), axis=1), axis=0)
            amax = tl.where(has_nan != 0, float("nan"), amax)
        """
        # TODO(nvfp4-v2): implement. See design doc §11.8 / §1.
        tl.static_assert(
            False, "_group_row_cast_col_rht_amax_kernel is not implemented"
        )

    @torch.library.custom_op(
        "torchao::triton_group_row_cast_col_rht_amax", mutates_args=()
    )
    def triton_group_row_cast_col_rht_amax(
        A: torch.Tensor,
        sign_vector: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-group RHT-128 columnwise amax and plain rowwise amax, dynamic signs.

        Args:
            A: packed ``(sum_M, N)`` bfloat16 tensor, row-major. N must be divisible
                by 128, and group offsets must be 128-row aligned so a token tile
                never straddles two groups.
            sign_vector: ``(128,)`` {-1, +1} device tensor. A live buffer updated in
                place by the cadence manager, so its address stays stable under
                CUDA-graph capture. Passing a tuple here is a ``TypeError``, by design.
            offsets: int32 cumulative row-end offsets, one per group, no leading zero.
            num_tensors: number of groups.
            packed_sequence_length: allocated row capacity of A.
            hidden_size: number of columns in A.
            shape_rep: SAME_BOTH_DIMS or VARYING_FIRST_DIM.
            logical_packed_length: one-element int32 CUDA tensor holding the valid
                padded row count, equal to ``offsets[-1]``. Rows beyond it are
                untouched capacity and are excluded from every group's amax.

        Returns:
            ``(amax_rht_a_t, amax_a)``, each ``(num_tensors,)`` float32 --
            **transformed first**, matching the shipped ``triton_group_rht_amax``:
              - ``amax_rht_a_t[g] = amax(abs(A_g.t() @ R_m))``
              - ``amax_a[g]       = amax(abs(A_g))``
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

        # Zero-initialized and reduced with atomic_max, so repeated launches are
        # idempotent and autotuning needs no reset_to_zero.
        row_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=A.device)
        col_amax = torch.zeros((num_tensors,), dtype=torch.float32, device=A.device)

        m, n = A.shape
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_cast_col_rht_amax_kernel[grid](
            A,
            B,
            offsets,
            row_amax,
            col_amax,
            m,
            n,
            num_tensors=num_tensors,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
            logical_packed_length_ptr=logical_packed_length,
        )
        return col_amax, row_amax

    @triton_group_row_cast_col_rht_amax.register_fake
    def _(
        A,
        sign_vector,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        logical_packed_length=None,
    ):
        col_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        row_amax = A.new_empty((num_tensors,), dtype=torch.float32)
        return col_amax, row_amax

else:

    def triton_group_row_cast_col_rht_amax(
        A: torch.Tensor,
        sign_vector: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_group_row_cast_col_rht_amax requires torch 2.10.0+ and triton "
            "installed"
        )
