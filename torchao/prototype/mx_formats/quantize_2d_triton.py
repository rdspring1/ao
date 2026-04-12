"""Triton kernel for 2D (16×16) NVFP4 E2M1 weight quantization."""

import itertools
import triton
import triton.language as tl
import torch
from torchao.prototype.mx_formats.hadamard_utils import (
    _compute_pid,
    convert_8xfp32_to_4xfp4_packed,
    _swizzle_scales,
)
from torchao.utils import is_sm_at_least_100


# SM100+ autotune configs. BLOCK_M=256 enables col TMA sf store; BLOCK_N=256 enables row TMA.
QUANTIZE_2D_CONFIGS: list[triton.Config] = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "NUM_STAGES": ns},
        num_warps=nw,
        num_stages=ns,
    )
    for bm, bn, ns, nw in itertools.product(
        [128, 256],
        [256],  # BLOCK_N: 256 enables rowwise TMA sf store
        [2, 3, 4],  # NUM_STAGES
        [4, 8],  # NUM_WARPS
    )
]


@triton.jit
def _nvfp4_2d_quantize(a, global_amax, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Compute per-16×16-block FP8 scale factors and scaled FP32 values for FP4 packing.

    Args:
        a: (BLOCK_M, BLOCK_N) bfloat16 tensor.
        global_amax: scalar float32 global amax.

    Returns:
        scale_inv: (BLOCK_M // 16, BLOCK_N // 16) float8e4nv per-block decode scales.
        scaled:    (BLOCK_M, BLOCK_N) float32 values scaled and clamped to FP4 range.
    """
    FP8_E4M3_MAX: tl.constexpr = 448.0
    FP4_E2M1_MAX: tl.constexpr = 6.0
    FP32_MAX: tl.constexpr = torch.finfo(torch.float32).max

    a_tile = tl.reshape(a, [BLOCK_M // 16, 16, BLOCK_N // 16, 16])
    abs_a_tile = tl.abs(a_tile)  # (BLOCK_M//16, 16, BLOCK_N//16, 16)
    tile_max = tl.max(
        abs_a_tile, axis=-1, keep_dims=True
    )  # (BLOCK_M//16, 16, BLOCK_N//16, 1)
    tile_max = tl.max(
        tile_max, axis=-3, keep_dims=True
    )  # (BLOCK_M//16, 1, BLOCK_N//16, 1)

    is_global_amax = global_amax == 0
    safe_global_amax = tl.where(is_global_amax, 1.0, global_amax)
    candidate = tl.minimum(FP8_E4M3_MAX * FP4_E2M1_MAX / safe_global_amax, FP32_MAX)
    candidate = tl.where(candidate == 0, 1.0, candidate)
    global_encode_scale = tl.where(is_global_amax, 1.0, candidate)
    global_decode_scale = 1.0 / global_encode_scale

    pvscale = (tile_max / FP4_E2M1_MAX) * global_encode_scale
    pvscale = tl.clamp(pvscale, -FP8_E4M3_MAX, FP8_E4M3_MAX)
    pvscale_fp8 = pvscale.to(tl.float8e4nv)
    scale_inv = tl.reshape(pvscale_fp8, [BLOCK_M // 16, BLOCK_N // 16])

    encode_scale = tl.minimum(
        1.0 / (pvscale_fp8.to(tl.float32) * global_decode_scale), FP32_MAX
    )

    scaled = a_tile * encode_scale
    scaled = tl.clamp(scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
    scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_N])
    return scale_inv, scaled


@triton.autotune(
    configs=QUANTIZE_2D_CONFIGS,
    key=["M", "N", "SWIZZLE_SCALE_FACTORS"],
    cache_results=True,
)
@triton.jit
def _weight_quantize_2d_kernel(
    a_ptr,
    out_ptr,
    scales_ptr,
    global_amax_ptr,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    SWIZZLE_SCALE_FACTORS: tl.constexpr,
):
    """2D (16×16) NVFP4 E2M1 weight quantization — one tile per CTA."""
    # Create TMA descriptors in-kernel from raw pointers, shape, and stride
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_M, BLOCK_N],
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[M, N // 2],
        strides=[N // 2, 1],
        block_shape=[BLOCK_M, BLOCK_N // 2],
    )
    if SWIZZLE_SCALE_FACTORS:
        sf_desc = tl.make_tensor_descriptor(
            scales_ptr,
            shape=[M // 128, N // 64, 32, 16],
            strides=[(N // 64) * 32 * 16, 32 * 16, 16, 1],
            block_shape=[BLOCK_M // 128, BLOCK_N // 64, 32, 16],
        )
    else:
        sf_desc = tl.make_tensor_descriptor(
            scales_ptr,
            shape=[M, N // 16],
            strides=[N // 16, 1],
            block_shape=[BLOCK_M, BLOCK_N // 16],
        )

    # Persistent grid-stride loop
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_N * num_pid_m
    num_tiles = num_pid_m * num_pid_n

    # Load global amax scalar once
    global_amax = tl.load(global_amax_ptr)

    for tile_id in tl.range(
        start_pid,
        num_tiles,
        NUM_SMS,
        flatten=False,
        num_stages=NUM_STAGES,
    ):
        pid_n, pid_m = _compute_pid(tile_id, num_pid_in_group, num_pid_n, GROUP_SIZE_N)

        # Load A (BLOCK_M, BLOCK_N)
        a = a_desc.load([pid_m * BLOCK_M, pid_n * BLOCK_N])

        # Compute per-16×16-block scales and scaled values
        scale_inv, scaled = _nvfp4_2d_quantize(a, global_amax, BLOCK_N, BLOCK_M)

        # Pack FP4 values into uint8 — non-transposed: (BLOCK_M, BLOCK_N//2, 2)
        scaled_pairs = scaled.reshape(BLOCK_M, BLOCK_N // 2, 2).split()
        scaled_fp4x2 = convert_8xfp32_to_4xfp4_packed(scaled_pairs)
        out_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N // 2], scaled_fp4x2)

        # Expand scales: (BLOCK_M//16, BLOCK_N//16) → (BLOCK_M, BLOCK_N//16)
        expand_sf = (
            tl.expand_dims(scale_inv, axis=1)
            .broadcast_to([BLOCK_M // 16, 16, BLOCK_N // 16])
            .reshape(BLOCK_M, BLOCK_N // 16)
        )
        if SWIZZLE_SCALE_FACTORS:
            swizzle_expand_sf = _swizzle_scales(expand_sf, BLOCK_M, BLOCK_N)
            sf_desc.store(
                [pid_m * BLOCK_M // 128, pid_n * BLOCK_N // 64, 0, 0], swizzle_expand_sf
            )
        else:
            sf_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N // 16], expand_sf)


def triton_weight_quantize_2d(
    A: torch.Tensor,
    swizzle_scale_factors: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2D (16×16) NVFP4 E2M1 weight quantization without RHT.

    Args:
        A:                     (M, N) bfloat16, row-major. M and N divisible by 16.
        swizzle_scale_factors: When True (requires M%128==0, N%128==0), returns scale
                               factors in SWIZZLE_32_4_4 layout (M//128, N//64, 32, 16).
                               When False, returns (M, N//16) contiguous row-major scales.

    Returns:
        Tuple of:
          - (M, N//2) uint8: packed FP4 E2M1 codes.
          - scale_factors float8_e4m3fn: per-block decode scale factors.
              swizzle_scale_factors=False: (M, N//16)
              swizzle_scale_factors=True:  (M//128, N//64, 32, 16)
          - scalar float32: global amax of A.
    """
    if not is_sm_at_least_100():
        raise NotImplementedError("triton_weight_quantize_2d requires SM100+")
    assert A.dtype == torch.bfloat16, f"Expected bfloat16, got {A.dtype}"
    assert A.ndim == 2, "Tensor A must be 2-D"
    assert A.is_contiguous(), "A must be row-major (contiguous)"
    M, N = A.shape
    assert M % 16 == 0, f"M must be divisible by 16, got M={M}"
    assert N % 16 == 0, f"N must be divisible by 16, got N={N}"

    global_amax = A.float().abs().max()
    out = torch.zeros((M, N // 2), dtype=torch.uint8, device=A.device)
    if swizzle_scale_factors:
        assert M % 128 == 0, f"M must be divisible by 128 for swizzling, got M={M}"
        assert N % 128 == 0, f"N must be divisible by 128 for swizzling, got N={N}"
        scale_factors = torch.empty(
            (M // 128, N // 64, 32, 16), dtype=torch.float8_e4m3fn, device=A.device
        )
    else:
        scale_factors = torch.empty(
            (M, N // 16), dtype=torch.float8_e4m3fn, device=A.device
        )

    NUM_SMS = torch.cuda.get_device_properties(A.device).multi_processor_count
    GROUP_SIZE_N: int = 8

    if hasattr(triton, "set_allocator"):
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device=A.device
            )
        )

    _weight_quantize_2d_kernel[(NUM_SMS,)](
        A,
        out,
        scale_factors,
        global_amax,
        M,
        N,
        GROUP_SIZE_N=GROUP_SIZE_N,
        NUM_SMS=NUM_SMS,
        SWIZZLE_SCALE_FACTORS=swizzle_scale_factors,
    )
    return out, scale_factors, global_amax
