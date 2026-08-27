# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped MS-EDEN quantize, RHT-128 on both axes.

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

   **Return order is rowwise first**, matching every sibling grouped quantize op in
   this directory: ``(row_fp4_rht_dy, row_sf_rht_dy, col_fp4_rht_dy_t,
   col_sf_rht_dy_t)``, the same shape as ``triton_group_rht_quantize_row_col``.

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

    @triton.jit
    def _ms_eden_correction_with_sr(sf,
                            fp4,
                            packed_fp4,
                            seed_base_ptr,
                            offset_base_ptr,
                            pid_m,
                            offs_n,
                            M,
                            BLOCK_M,
                            BLOCK_N):
        # Get MS-EDEN correction factor: dot(v, q) / dot(q, q) where v is the
        # original fp32 values scaled and clamped to fp4 range but not quantized
        # to its fp4 codebook. q is the dequantized packed fp4 values, so it is
        # the fp32 version of the quantized fp4 codes. 
        fp4_dequant = convert_4xfp4_packed_to_8xfp32(packed_fp4)
        fp4_blocks = fp4.reshape(BLOCK_M, BLOCK_N // 16, 16)
        fp4_dequant_blocks = fp4_dequant.reshape(BLOCK_M, BLOCK_N // 16, 16)
        dot_sq = tl.sum(fp4_blocks * fp4_blocks, axis=-1)
        dot_cross = tl.sum(fp4_blocks * fp4_dequant_blocks, axis=-1)
        ratio = dot_sq / dot_cross
        is_finite = tl.abs(ratio) < float("inf")
        correction = tl.where((dot_cross != 0.0) & is_finite, ratio, 1.0)

        # Apply stochastic rounding and MS-EDEN correction to the block scale.
        corrected_sf = sf * correction
        sr_corrected_sf = _stochastic_rounding_fp8_e4m3(
            corrected_sf,
            BLOCK_M,
            BLOCK_N,
            seed_base_ptr,
            offset_base_ptr,
            pid_m,
            offs_n,
            M,
        )
        return sr_corrected_sf

    @triton.jit
    def _stochastic_rounding_fp8_e4m3(
        x,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        seed_ptr,
        offset_base_ptr,
        pid_m,
        offs_n,
        M,
    ):
        seed = tl.load(seed_ptr)
        offset_base = tl.load(offset_base_ptr).to(tl.uint64) & 0xFFFFFFFF

        # Generate a index tile
        BLOCK_M_SF: tl.constexpr = BLOCK_M // 16
        packed_inner_t = pid_m * BLOCK_M_SF + tl.arange(0, BLOCK_M_SF)
        linear_idx = offs_n[:, None] * (M // 16) + packed_inner_t[None, :]

        offset = (linear_idx.to(tl.uint64) << 32) | offset_base
        rbits = tl.randint(seed, offset).to(tl.uint32)

        r = x * (2.0**-120)
        r_int = r.to(tl.uint32, bitcast=True)
        u = r_int + (rbits & ((1 << 20) - 1))
        u = (u >> 20) << 20
        u_float = u.to(tl.float32, bitcast=True)
        return u_float * (2.0**120)

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

                # --- rowwise / dgrad, returned first per the return contract ---
                a_rht = tl.dot(<reshaped a>, <row_rht>)                   # R_n
                row_sf, row_scaled = _quantize_ms_eden(
                    a_rht, tl.load(amax_row_rht_ptr + group_idx),
                    BLOCK_M, BLOCK_N, row_seed_base_ptr, row_offset_base_ptr, tile_idx)
                ...
                # --- columnwise / wgrad ---
                a_t_rht = tl.dot(<reshaped trans(a)>, <col_rht>)          # R_m
                col_sf, col_scaled = _quantize_ms_eden(
                    a_t_rht, tl.load(amax_col_rht_ptr + group_idx),
                    BLOCK_N, BLOCK_M, col_seed_base_ptr, col_offset_base_ptr, tile_idx)

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
        # The RHT runs along the inner axis of the reshape below, so each
        # (BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE) chunk must stay inside one row
        # of  a_t -- i.e. inside one hidden column's run of BLOCK_M tokens. At
        # BLOCK_M % RHT_SIZE != 0 a chunk straddles two hidden columns and the
        # transform is applied across the wrong axis
        tl.static_assert(
            BLOCK_M % RHT_SIZE == 0, "columnwise RHT requires BLOCK_M % RHT_SIZE == 0"
        )
        VARYING_FIRST_DIM: tl.constexpr = 1

        tile_idx = tl.program_id(0)
        num_tiles_token = tl.cdiv(M, BLOCK_M)
        num_tiles_hidden = tl.cdiv(N, BLOCK_N)
        # int64 tile indices: the A load (offs_m * N) and the packed-store offsets
        # (outer * N//2, outer_t * M//2) overflow int32 once the packed token count
        # times N exceeds 2**31, silently wrapping to bad addresses.
        pid_m = (tile_idx // num_tiles_hidden).to(tl.int64)
        pid_n = tile_idx - pid_m * num_tiles_hidden
        token_offset = pid_m * BLOCK_M
        logical_packed_length = tl.load(logical_packed_length_ptr)

        if token_offset < logical_packed_length:
            if SHAPE_REP == VARYING_FIRST_DIM:
                group_idx = _get_group_idx_binary(
                    token_offset,
                    offsets_ptr,
                    num_tensors,
                )
            else:
                group_idx = pid_m // (num_tiles_token // num_tensors)

            offs_m = token_offset + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            a = tl.load(a_ptr + offs_m[:, None] * N + offs_n[None, :])

            rht_offsets = (
                tl.arange(0, RHT_SIZE)[:, None] * RHT_SIZE
                + tl.arange(0, RHT_SIZE)[None, :]
            )
            row_rht = tl.load(row_rht_ptr + rht_offsets)
            col_rht = tl.load(col_rht_ptr + rht_offsets)

            row_global_amax = tl.load(amax_row_rht_ptr + group_idx)
            col_global_amax = tl.load(amax_col_rht_ptr + group_idx)

            a_reshape = tl.reshape(a, [BLOCK_M * BLOCK_N // RHT_SIZE, RHT_SIZE])
            a_rht = tl.dot(a_reshape, row_rht)

            a_t = tl.trans(a)
            a_t_reshape = tl.reshape(a_t, [BLOCK_N * BLOCK_M // RHT_SIZE, RHT_SIZE])
            a_t_rht = tl.dot(a_t_reshape, col_rht)

            # TE's fast math consumes the fp32 accumulator directly, so the bfloat16
            # round-through is exact-mode only.
            if not FAST_MATH:
                a_rht = a_rht.to(tl.bfloat16)
                a_t_rht = a_t_rht.to(tl.bfloat16)

            col_block_amax = tl.max(
                tl.abs(tl.reshape(a_t_rht, [BLOCK_N, BLOCK_M // 16, 16])), axis=-1
            )
            col_scale, col_scaled = _nvfp4_quantize(
                a_t_rht,
                col_block_amax,
                col_global_amax,
                BLOCK_N,
                BLOCK_M,
                FAST_MATH,
                256.0,  # FP8_E4M3_MAX
            )

            col_scaled_pairs = col_scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
            col_fp4 = convert_8xfp32_to_4xfp4_packed(col_scaled_pairs)
            packed_inner_t = pid_m * (BLOCK_M // 2) + tl.arange(0, BLOCK_M // 2)
            packed_offsets_t = offs_n[:, None] * (M // 2) + packed_inner_t[None, :]
            tl.store(qa_t_ptr + packed_offsets_t, col_fp4)

            # Apply ms_eden correction to col_scale after packed fp4 codes created
            sr_corrected_col_scale = _ms_eden_correction_with_sr(
                col_scale,
                col_scaled,
                col_fp4,
                col_seed_base_ptr,
                col_offset_base_ptr,
                pid_n,
                offs_m,
                N,
                BLOCK_N,
                BLOCK_M,
            )

            col_swizzled = _swizzle_scales(sr_corrected_col_scale, BLOCK_N, BLOCK_M)
            # Columnwise puts the grouped token axis on the inner (64-blocked) side,
            # so the tiling restarts per group. The rowwise store below has it on
            # the outer axis, where a group is already contiguous.
            _store_grouped_scales_swizzle(
                col_swizzled,
                sfa_t_ptr,
                pid_n,
                pid_m,
                offsets_ptr,
                group_idx,
                N,
                BLOCK_N,
                BLOCK_M,
            )

            row_block_amax = tl.max(
                tl.abs(tl.reshape(a_rht, [BLOCK_N, BLOCK_M // 16, 16])), axis=-1
            )
            row_scale, row_scaled = _nvfp4_quantize(
                a_rht,
                row_block_amax,
                row_global_amax,
                BLOCK_M,
                BLOCK_N,
                FAST_MATH,
                256.0,  # FP8_E4M3_MAX
            )
            row_scaled_pairs = row_scaled.reshape(BLOCK_N, BLOCK_M // 2, 2).split()
            row_fp4 = convert_8xfp32_to_4xfp4_packed(col_scaled_pairs)

            # Store the packed fp4 codes
            outer = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            packed_inner = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
            packed_offsets = outer[:, None] * (N // 2) + packed_inner[None, :]
            tl.store(qa_ptr + packed_offsets, row_fp4)

            # Apply ms_eden correction to row_scale after packed fp4 codes created
            sr_corrected_row_scale = _ms_eden_correction_with_sr(
                row_scale,
                row_scaled,
                row_fp4,
                row_seed_base_ptr,
                row_offset_base_ptr,
                pid_m,
                offs_n,
                M,
                BLOCK_M,
                BLOCK_N,
            )

            row_swizzled = _swizzle_scales(row_scale, BLOCK_M, BLOCK_N)
            _store_scales_swizzle(
                row_swizzled,
                sfa_ptr,
                pid_m,
                pid_n,
                M,
                N,
                BLOCK_M,
                BLOCK_N,
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
            ``(row_fp4_rht_dy, row_sf_rht_dy, col_fp4_rht_dy_t, col_sf_rht_dy_t)`` --
            **rowwise first**, like every sibling grouped op; see the module
            docstring warning for why this now diverges from design doc §6/§11.3.
              - ``(sum_M, N//2)`` uint8 rowwise codes.
              - ``(sum_M, N//16)`` float8_e4m3fn rowwise scales.
              - ``(N, sum_M//2)`` uint8 columnwise transposed codes.
              - ``(N, sum_M//16)`` float8_e4m3fn columnwise scales.

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
        # Rowwise pair first, matching every sibling quantize op.
        return qa_base, sfa_return, qd, sfd_return

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
        return qa_base, sfa, qd, sfd

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
