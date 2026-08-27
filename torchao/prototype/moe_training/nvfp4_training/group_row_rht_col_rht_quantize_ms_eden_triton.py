# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped MS-EDEN quantize, RHT-128 on both axes. Design doc §11.3 (§6).

The V2 backward gradient quantize, producing both operands from one read of ``dy``:

    row_fp4_rht_dy   = ms_eden(dy_g @ R_n)       # dgrad operand
    col_fp4_rht_dy_t = ms_eden(dy_g.t() @ R_m)   # wgrad operand

MS-EDEN replaces FP4 stochastic rounding with RTNE codes plus a corrected,
**stochastically-rounded E4M3 block scale** -- the scale is the only random step.
The block-scale ceiling is 256, not 448, leaving headroom for the correction. That
is why these operands decode with numerator ``256 * 6 = 1536`` while every cast
operand uses ``448 * 6 = 2688``. Pairing an MS-EDEN operand with the cast numerator
leaves the forward correct and fails backward by roughly 40%.

.. warning::

   **Return order is columnwise first**, matching design doc §6/§11.3:
   ``(col_fp4_rht_dy_t, col_sf_rht_dy_t, row_fp4_rht_dy, row_sf_rht_dy)``.

   This is the opposite of every sibling grouped quantize op in this directory --
   ``triton_group_rht_quantize_row_col`` returns rowwise first. The order here follows
   the design doc because that is the contract the recipe is checked against. A swapped
   unpack is only caught by a shape error when ``M != N``; on a square layer it corrupts
   silently.

Device helpers this kernel needs but torchao does not yet have -- port them from
``cutile/nvfp4_v2_triton/kernels/hadamard_utils.py`` in the monorepo:

* ``stochastic_rounding_fp8_e4m3`` (monorepo hadamard_utils.py:249)
* ``_quantize_ms_eden`` (monorepo hadamard_utils.py:638)

They are the MS-EDEN algorithm itself rather than shared plumbing, so they belong
with the kernel body.

A dense linear is the degenerate ``num_tensors = 1`` case: ``offsets = [M]``.
"""

from typing import Optional

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

from .group_hadamard_utils import _validate_graph_amax, _validate_rng_state

RHT_SIZE = 128

# MS-EDEN's block-scale ceiling. Lower than the 448.0 a plain NVFP4 cast uses, to
# leave headroom for the stochastically-rounded scale correction.
EDEN_BLOCK_SCALE_MAX = 256.0

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
        _nvfp4_global_scales,
        _store_grouped_scales_swizzle,
        _store_scales_swizzle,
        _swizzle_scales,
        convert_8xfp32_to_4xfp4_packed,
        get_dynamic_rht_matrix,
    )

    # num_warps pinned at 4 for the same reason as the other grouped quantize
    # kernels: the body is register-heavy and the 8-warp win is M-dependent, while
    # M is dropped from the key so one config is cached across all M.
    _GROUP_MS_EDEN_CONFIGS: list[triton.Config] = [
        triton.Config({}, num_warps=4, num_stages=ns) for ns in (2, 3, 4)
    ]

    @triton.autotune(configs=_GROUP_MS_EDEN_CONFIGS, key=["N"])
    @triton.jit
    def _group_row_rht_col_rht_quantize_ms_eden_kernel(
        a_ptr,
        row_rht_ptr,
        col_rht_ptr,
        qa_ptr,
        sfa_ptr,
        qa_t_ptr,
        sfa_t_ptr,
        offsets_ptr,
        amax_row_rht_ptr,
        amax_col_rht_ptr,
        col_seed_base_ptr,
        col_offset_base_ptr,
        row_seed_base_ptr,
        row_offset_base_ptr,
        M,
        N,
        num_tensors: tl.constexpr,
        SHAPE_REP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RHT_SIZE: tl.constexpr,
        logical_packed_length_ptr,
    ):
        """Grouped MS-EDEN quantization of ``dy @ R_n`` and ``dy.t() @ R_m``.

        Flat grid ``(cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),)``. Sketch:

            token_tile_idx = (tile_idx // num_tiles_hidden).to(tl.int64)   # int64!
            token_offset = token_tile_idx * BLOCK_M
            if token_offset < tl.load(logical_packed_length_ptr):
                group_idx = _get_group_idx_binary(token_offset, offsets_ptr, num_tensors)
                a = <load (BLOCK_M, BLOCK_N) tile>

                # --- columnwise / wgrad, stored first per the return contract ---
                a_t_rht = tl.dot(<reshaped trans(a)>, <col_rht>)          # R_m
                col_sf, col_scaled = _quantize_ms_eden(
                    a_t_rht, tl.load(amax_col_rht_ptr + group_idx),
                    BLOCK_N, BLOCK_M, col_seed_base_ptr, col_offset_base_ptr, tile_idx)
                ...
                # --- rowwise / dgrad ---
                a_rht = tl.dot(<reshaped a>, <row_rht>)                   # R_n
                row_sf, row_scaled = _quantize_ms_eden(
                    a_rht, tl.load(amax_row_rht_ptr + group_idx),
                    BLOCK_M, BLOCK_N, row_seed_base_ptr, row_offset_base_ptr, tile_idx)

        MS-EDEN specifics the body must honor:

        * The global scales come from ``_nvfp4_global_scales(amax, 256.0)``, not
          ``448.0``. This is what makes the operand's decode numerator 1536.
        * Codes are RTNE (``convert_8xfp32_to_4xfp4_packed``); randomness enters only
          through the E4M3 block scale.
        * The correction factor is ``dot(v, q) / dot(q, q)``-shaped. Clamp it to 1.0
          when ``dot_cross`` is zero or the ratio is non-finite -- an exactly
          representable block must correct by exactly 1.0.
        * Row and column streams draw from independent Philox counters.

        The columnwise scale store is the one place grouping changes the layout: each
        group owns a separately swizzled ``(hidden, group_tokens//16)`` block and the
        allocation is their flat concatenation. **Compute that per-group word offset
        in 64-bit** -- at ``hidden = 7168`` a 32-bit product wraps past roughly 300k
        rows. ``_store_grouped_scales_swizzle`` already handles this.

        Rows at or beyond ``logical_packed_length`` must be left zero-filled.
        """
        # TODO(nvfp4-v2): implement. See design doc §11.3 / §6.
        tl.static_assert(
            False,
            "_group_row_rht_col_rht_quantize_ms_eden_kernel is not implemented",
        )

    @torch.library.custom_op(
        "torchao::triton_group_row_rht_col_rht_quantize_ms_eden", mutates_args=()
    )
    def triton_group_row_rht_col_rht_quantize_ms_eden(
        dy: torch.Tensor,
        amax_rht_dy: torch.Tensor,
        amax_rht_dy_t: torch.Tensor,
        dgrad_rht: torch.Tensor,
        wgrad_rht: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        rng_state: torch.Tensor,
        logical_packed_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-group MS-EDEN quantization of the rotated gradient and its transpose.

        Args:
            dy: packed ``(sum_M, N)`` bfloat16 gradient, row-major.
            amax_rht_dy: ``(num_tensors,)`` float32, first return of
                ``triton_group_row_rht_col_rht_amax``.
            amax_rht_dy_t: ``(num_tensors,)`` float32, second return of the same op.
            dgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the row axis.
            wgrad_rht: ``(128,)`` {-1, +1} device tensor rotating the transposed axis.
            offsets: int32 cumulative row-end offsets, one per group.
            num_tensors: number of groups.
            packed_sequence_length: allocated row capacity of dy.
            hidden_size: number of columns in dy.
            shape_rep: SAME_BOTH_DIMS or VARYING_FIRST_DIM.
            rng_state: int64 CUDA tensor ``[col_seed, col_offset, row_seed,
                row_offset]``. Always required -- MS-EDEN is never deterministic, so
                unlike the cast quantizers there is no RTNE-only mode to fall back to.
            logical_packed_length: one-element int32 CUDA tensor, ``offsets[-1]``.

        Returns:
            ``(col_fp4_rht_dy_t, col_sf_rht_dy_t, row_fp4_rht_dy, row_sf_rht_dy)`` --
            **columnwise first**; see the module docstring warning.
              - ``(N, sum_M//2)`` uint8 columnwise transposed codes.
              - ``(N, sum_M//16)`` float8_e4m3fn columnwise scales.
              - ``(sum_M, N//2)`` uint8 rowwise codes.
              - ``(sum_M, N//16)`` float8_e4m3fn rowwise scales.

            Decode both with ``EDEN_NUMERATOR`` (1536), never 2688.
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
        row_amax = _validate_graph_amax(
            amax_rht_dy, "amax_rht_dy", num_tensors, dy.device
        )
        col_amax = _validate_graph_amax(
            amax_rht_dy_t, "amax_rht_dy_t", num_tensors, dy.device
        )
        # MS-EDEN always draws, so SR is unconditionally on for validation purposes.
        rng_state = _validate_rng_state(rng_state, dy.device, True)

        qa_base = torch.empty(
            (packed_sequence_length, hidden_size // 2),
            dtype=torch.uint8,
            device=dy.device,
        )
        sfa_storage = torch.empty(
            (packed_sequence_length // 128, hidden_size // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=dy.device,
        )
        sfa_return = sfa_storage.view(packed_sequence_length, hidden_size // 16)

        qd = torch.empty(
            (hidden_size, packed_sequence_length // 2),
            dtype=torch.uint8,
            device=dy.device,
        )
        sfd_storage = torch.empty(
            (hidden_size // 128, packed_sequence_length // 64, 32, 16),
            dtype=torch.float8_e4m3fn,
            device=dy.device,
        )
        sfd_return = sfd_storage.view(hidden_size, packed_sequence_length // 16)

        m, n = dy.shape
        if logical_packed_length is None:
            logical_packed_length = offsets[-1:]
        grid = (triton.cdiv(m, BLOCK_M) * triton.cdiv(n, BLOCK_N),)
        _group_row_rht_col_rht_quantize_ms_eden_kernel[grid](
            dy,
            row_rht,
            col_rht,
            qa_base,
            sfa_storage,
            qd,
            sfd_storage,
            offsets,
            row_amax,
            col_amax,
            rng_state[0:1],
            rng_state[1:2],
            rng_state[2:3],
            rng_state[3:4],
            m,
            n,
            num_tensors=num_tensors,
            SHAPE_REP=shape_rep,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            RHT_SIZE=RHT_SIZE,
            logical_packed_length_ptr=logical_packed_length,
        )
        # Columnwise pair first, per design doc §6/§11.3.
        return qd, sfd_return, qa_base, sfa_return

    @triton_group_row_rht_col_rht_quantize_ms_eden.register_fake
    def _(
        dy,
        amax_rht_dy,
        amax_rht_dy_t,
        dgrad_rht,
        wgrad_rht,
        offsets,
        num_tensors,
        packed_sequence_length,
        hidden_size,
        shape_rep,
        rng_state,
        logical_packed_length=None,
    ):
        qd = dy.new_empty((hidden_size, packed_sequence_length // 2), dtype=torch.uint8)
        sfd = dy.new_empty(
            (hidden_size, packed_sequence_length // 16), dtype=torch.float8_e4m3fn
        )
        qa_base = dy.new_empty(
            (packed_sequence_length, hidden_size // 2), dtype=torch.uint8
        )
        sfa = dy.new_empty(
            (packed_sequence_length, hidden_size // 16), dtype=torch.float8_e4m3fn
        )
        return qd, sfd, qa_base, sfa

else:

    def triton_group_row_rht_col_rht_quantize_ms_eden(
        dy: torch.Tensor,
        amax_rht_dy: torch.Tensor,
        amax_rht_dy_t: torch.Tensor,
        dgrad_rht: torch.Tensor,
        wgrad_rht: torch.Tensor,
        offsets: torch.Tensor,
        num_tensors: int,
        packed_sequence_length: int,
        hidden_size: int,
        shape_rep: int,
        rng_state: torch.Tensor,
        logical_packed_length: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError(
            "triton_group_row_rht_col_rht_quantize_ms_eden requires torch 2.10.0+ "
            "and triton installed"
        )
