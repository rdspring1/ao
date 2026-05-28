"""Triton kernel for 2D (16×16) NVFP4 E2M1 weight quantization."""

import torch
from torch.utils._triton import has_triton

from torchao.utils import torch_version_at_least

if torch_version_at_least("2.10.0") and has_triton():
    from typing import Tuple

    import triton
    import triton.language as tl

    from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
        _swizzle_scales,
        prepare_for_cuda_graph,
    )
    from torchao.utils import is_sm_at_least_100

    # TE-style wave tile: one CTA owns a 128x128 chunk and processes it as
    # four 32x128 panels.
    QUANTIZE_2D_CONFIGS: list[triton.Config] = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=2)
    ]

    @triton.jit
    def _nvfp4_2d_scale_factors(
        tile_max, global_amax, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
    ):
        """Compute per-16x16-block FP8 scale factors and FP4 encode scales.

        Args:
            tile_max: (BLOCK_M//16, 1, BLOCK_N//16, 1) max abs value per 16x16 tile.
            global_amax: scalar float32 global amax.

        Returns:
            scale_inv: (BLOCK_M // 16, BLOCK_N // 16) float8e4nv per-block decode scales.
            encode_scale: (BLOCK_M//16, 1, BLOCK_N//16, 1) float32 per-block encode scales.
        """
        FP8_E4M3_EPS: tl.constexpr = torch.finfo(torch.float8_e4m3fn).tiny
        FP8_E4M3_MAX: tl.constexpr = 448.0
        FP4_E2M1_MAX: tl.constexpr = 6.0
        FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

        is_global_amax = global_amax == 0
        safe_global_amax = tl.where(is_global_amax, 1.0, global_amax)
        candidate = tl.minimum(FP8_E4M3_MAX * FP4_E2M1_MAX / safe_global_amax, FP32_MAX)
        candidate = tl.where(candidate == 0, 1.0, candidate)
        global_encode_scale = tl.where(is_global_amax, 1.0, candidate)
        global_decode_scale = 1.0 / global_encode_scale

        pvscale = (tile_max / FP4_E2M1_MAX) * global_encode_scale
        pvscale = tl.clamp(pvscale, FP8_E4M3_EPS, FP8_E4M3_MAX)
        pvscale_fp8 = pvscale.to(tl.float8e4nv)
        scale_inv = tl.reshape(pvscale_fp8, [BLOCK_M // 16, BLOCK_N // 16])

        encode_scale = tl.minimum(
            1.0 / (pvscale_fp8.to(tl.float32) * global_decode_scale), FP32_MAX
        )
        return scale_inv, encode_scale

    @triton.jit
    def _convert_8xbf16_scaled_to_4xfp4_packed(x_pairs, scale):
        x_fp4x2 = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b16 e0, e1, e2, e3, o0, o1, o2, o3;
            .reg .b32 fe0, fe1, fe2, fe3, fo0, fo1, fo2, fo3;
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {e0, e1}, $1;
            mov.b32 {e2, e3}, $2;
            mov.b32 {o0, o1}, $3;
            mov.b32 {o2, o3}, $4;
            cvt.f32.bf16 fe0, e0;
            cvt.f32.bf16 fe1, e1;
            cvt.f32.bf16 fe2, e2;
            cvt.f32.bf16 fe3, e3;
            cvt.f32.bf16 fo0, o0;
            cvt.f32.bf16 fo1, o1;
            cvt.f32.bf16 fo2, o2;
            cvt.f32.bf16 fo3, o3;
            mul.f32 fe0, fe0, $5;
            mul.f32 fo0, fo0, $5;
            mul.f32 fe1, fe1, $6;
            mul.f32 fo1, fo1, $6;
            mul.f32 fe2, fe2, $7;
            mul.f32 fo2, fo2, $7;
            mul.f32 fe3, fe3, $8;
            mul.f32 fo3, fo3, $8;
            cvt.rn.satfinite.e2m1x2.f32 byte0, fo0, fe0;
            cvt.rn.satfinite.e2m1x2.f32 byte1, fo1, fe1;
            cvt.rn.satfinite.e2m1x2.f32 byte2, fo2, fe2;
            cvt.rn.satfinite.e2m1x2.f32 byte3, fo3, fe3;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[x_pairs[0], x_pairs[1], scale],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )
        return x_fp4x2

    @triton.jit
    def _pack_fp4_scaled_at(
        a_tile,
        encode_scale,
        out_desc,
        offset_m,
        offset_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        a_pairs = tl.reshape(a_tile, [BLOCK_M, BLOCK_N // 2, 2]).split()
        scale_pairs = encode_scale.broadcast_to(
            [BLOCK_M // 16, 16, BLOCK_N // 16, 8]
        ).reshape(BLOCK_M, BLOCK_N // 2)
        scaled_fp4x2 = _convert_8xbf16_scaled_to_4xfp4_packed(a_pairs, scale_pairs)
        out_desc.store([offset_m, offset_n], scaled_fp4x2)

    @triton.jit
    def _swizzle_scale_factors(
        scale_inv,
        sf_desc,
        pid_outer,
        pid_inner,
        BLOCK_OUTER: tl.constexpr,
        BLOCK_INNER: tl.constexpr,
    ):
        expand_sf = (
            tl.expand_dims(scale_inv, axis=1)
            .broadcast_to([BLOCK_OUTER // 16, 16, BLOCK_INNER // 16])
            .reshape(BLOCK_OUTER, BLOCK_INNER // 16)
        )
        swizzle_expand_sf = _swizzle_scales(expand_sf, BLOCK_OUTER, BLOCK_INNER)
        sf_desc.store(
            [
                pid_outer * BLOCK_OUTER // 128,
                pid_inner * BLOCK_INNER // 64,
                0,
                0,
            ],
            swizzle_expand_sf,
        )

    @triton.jit
    def _process_stage_32x128(
        a_desc,
        a_fp4_desc,
        a_t_fp4_desc,
        pid_m,
        pid_n,
        stage: tl.constexpr,
        global_amax,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        STAGE_M: tl.constexpr,
    ):
        stage_m = pid_m * BLOCK_M + stage * STAGE_M
        tile_n = pid_n * BLOCK_N

        a = a_desc.load([stage_m, tile_n])
        a_tile = tl.reshape(a, [STAGE_M // 16, 16, BLOCK_N // 16, 16])

        abs_a_tile = tl.abs(a_tile)
        tile_max = tl.max(abs_a_tile, axis=-1, keep_dims=True)
        tile_max = tl.max(tile_max, axis=-3, keep_dims=True)

        a_sf, encode_scale = _nvfp4_2d_scale_factors(
            tile_max, global_amax, STAGE_M, BLOCK_N
        )
        _pack_fp4_scaled_at(
            a_tile,
            encode_scale,
            a_fp4_desc,
            stage_m,
            pid_n * BLOCK_N // 2,
            STAGE_M,
            BLOCK_N,
        )

        a_t_tile = tl.permute(a_tile, [2, 3, 0, 1])
        encode_scale_t = tl.permute(encode_scale, [2, 3, 0, 1])
        a_t_sf = tl.reshape(
            tl.permute(
                tl.reshape(a_sf, [STAGE_M // 16, 1, BLOCK_N // 16, 1]),
                [2, 3, 0, 1],
            ),
            [BLOCK_N // 16, STAGE_M // 16],
        )
        _pack_fp4_scaled_at(
            a_t_tile,
            encode_scale_t,
            a_t_fp4_desc,
            pid_n * BLOCK_N,
            pid_m * BLOCK_M // 2 + stage * STAGE_M // 2,
            BLOCK_N,
            STAGE_M,
        )

        return a_sf, a_t_sf

    @triton.autotune(
        configs=QUANTIZE_2D_CONFIGS,
        key=["M", "N"],
        cache_results=True,
    )
    @triton.jit
    def triton_quantize_2d_weight(
        a_ptr,
        out_ptr,
        scales_ptr,
        a_t_fp4_ptr,
        a_t_sf_ptr,
        global_amax_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """2D (16×16) NVFP4 E2M1 weight quantization — one tile per CTA."""
        STAGE_M: tl.constexpr = 32

        # Create TMA descriptors in-kernel from raw pointers, shape, and stride
        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[STAGE_M, BLOCK_N],
        )
        out_desc = tl.make_tensor_descriptor(
            out_ptr,
            shape=[M, N // 2],
            strides=[N // 2, 1],
            block_shape=[STAGE_M, BLOCK_N // 2],
        )
        sf_desc = tl.make_tensor_descriptor(
            scales_ptr,
            shape=[M // 128, N // 64, 32, 16],
            strides=[(N // 64) * 32 * 16, 32 * 16, 16, 1],
            block_shape=[BLOCK_M // 128, BLOCK_N // 64, 32, 16],
        )
        a_t_fp4_desc = tl.make_tensor_descriptor(
            a_t_fp4_ptr,
            shape=[N, M // 2],
            strides=[M // 2, 1],
            block_shape=[BLOCK_N, STAGE_M // 2],
        )
        a_t_sf_desc = tl.make_tensor_descriptor(
            a_t_sf_ptr,
            shape=[N // 128, M // 64, 32, 16],
            strides=[(M // 64) * 32 * 16, 32 * 16, 16, 1],
            block_shape=[BLOCK_N // 128, BLOCK_M // 64, 32, 16],
        )

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        # Load global amax scalar once
        global_amax = tl.load(global_amax_ptr)

        # Match TE's chunk structure: a 128x128 CTA is processed as four
        # 32x128 panels, while scale tensors are concatenated and swizzled once.
        a_sf0, a_t_sf0 = _process_stage_32x128(
            a_desc,
            out_desc,
            a_t_fp4_desc,
            pid_m,
            pid_n,
            0,
            global_amax,
            BLOCK_M,
            BLOCK_N,
            STAGE_M,
        )
        a_sf1, a_t_sf1 = _process_stage_32x128(
            a_desc,
            out_desc,
            a_t_fp4_desc,
            pid_m,
            pid_n,
            1,
            global_amax,
            BLOCK_M,
            BLOCK_N,
            STAGE_M,
        )
        a_sf2, a_t_sf2 = _process_stage_32x128(
            a_desc,
            out_desc,
            a_t_fp4_desc,
            pid_m,
            pid_n,
            2,
            global_amax,
            BLOCK_M,
            BLOCK_N,
            STAGE_M,
        )
        a_sf3, a_t_sf3 = _process_stage_32x128(
            a_desc,
            out_desc,
            a_t_fp4_desc,
            pid_m,
            pid_n,
            3,
            global_amax,
            BLOCK_M,
            BLOCK_N,
            STAGE_M,
        )

        a_sf = tl.cat(tl.cat(a_sf0, a_sf1, dim=0), tl.cat(a_sf2, a_sf3, dim=0), dim=0)
        _swizzle_scale_factors(a_sf, sf_desc, pid_m, pid_n, BLOCK_M, BLOCK_N)

        a_t_sf = tl.cat(
            tl.cat(a_t_sf0, a_t_sf1, dim=1),
            tl.cat(a_t_sf2, a_t_sf3, dim=1),
            dim=1,
        )
        _swizzle_scale_factors(a_t_sf, a_t_sf_desc, pid_n, pid_m, BLOCK_N, BLOCK_M)

    @torch.library.custom_op("torchao::triton_weight_quantize_2d", mutates_args=())
    def triton_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """2D (16×16) NVFP4 E2M1 weight quantization without RHT.

        Args:
            A:           (M, N) bfloat16, row-major. M and N divisible by 16.
            global_amax: scalar float32 global absolute maximum of A. Caller computes
                         ``A.float().abs().max()`` (and optionally all-reduces for TP)
                         before passing in.

        Returns:
            4-tuple of:
              - (M, N//2) uint8: rowwise FP4 codes.
              - (M//128, N//64, 32, 16) float8_e4m3fn: rowwise swizzled scale factors.
              - (N, M//2) uint8: colwise FP4 codes (rowwise W.T).
              - (N//128, M//64, 32, 16) float8_e4m3fn: colwise swizzled scale factors.
        """
        if not is_sm_at_least_100():
            raise NotImplementedError("triton_weight_quantize_2d requires SM100+")
        if A.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16, got {A.dtype}")
        if A.ndim != 2:
            raise ValueError("Tensor A must be 2-D")
        if not A.is_contiguous():
            raise ValueError("A must be row-major (contiguous)")
        M, N = A.shape
        if M % 16 != 0:
            raise ValueError(f"M must be divisible by 16, got M={M}")
        if N % 16 != 0:
            raise ValueError(f"N must be divisible by 16, got N={N}")
        if M % 128 != 0:
            raise ValueError(f"M must be divisible by 128 for swizzling, got M={M}")
        if N % 128 != 0:
            raise ValueError(f"N must be divisible by 128 for swizzling, got N={N}")

        if hasattr(triton, "set_allocator"):
            _ws_nbytes = max(
                131072,
                triton.cdiv(M, 128) * triton.cdiv(N, 128) * 640,
            )
            _ws = prepare_for_cuda_graph(A.device, nbytes=_ws_nbytes)
            triton.set_allocator(lambda size, align, stream: _ws[: max(size, 1)])

        a_fp4 = torch.zeros((M, N // 2), dtype=torch.uint8, device=A.device)
        a_sf = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        a_t_fp4 = torch.zeros((N, M // 2), dtype=torch.uint8, device=A.device)
        a_t_sf = torch.empty(
            (N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )

        grid = lambda meta: (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

        try:
            triton_quantize_2d_weight[grid](
                A,
                a_fp4,
                a_sf,
                a_t_fp4,
                a_t_sf,
                global_amax,
                M,
                N,
            )
        finally:
            if hasattr(triton, "set_allocator"):
                triton.set_allocator(None)
        return a_fp4, a_sf, a_t_fp4, a_t_sf

    @triton_weight_quantize_2d.register_fake
    def _(A, global_amax):
        M, N = A.shape
        codes = A.new_empty((M, N // 2), dtype=torch.uint8)
        sf = A.new_empty((M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn)
        t_codes = A.new_empty((N, M // 2), dtype=torch.uint8)
        t_sf = A.new_empty((N // 128, M // 64, 32, 16), dtype=torch.float8_e4m3fn)
        return codes, sf, t_codes, t_sf

else:

    def triton_weight_quantize_2d(
        A: torch.Tensor,
        global_amax: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "triton_weight_quantize_2d requires torch 2.10.0+ and triton installed"
        )
