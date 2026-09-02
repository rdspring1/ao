"""CuTe DSL kernels for kitchen's MS-EDEN NVFP4 QDQ (recipe 100483).

Bitwise identical to :mod:`eden_reference`, which transcribes
``quantize_transpose_vector_blockwise_fp4_eden.cu``. The device helpers that are
genuinely shared with the 9004 path -- the E4M3 RNE grid, the E2M1 RNE grid, the
``torch.sign`` zero convention and the Philox round function -- are imported from
:mod:`nvfp4_cutedsl` rather than restated, so the two recipes cannot drift apart
on the arithmetic they have in common.

What is *not* shared, and is why this is a separate kernel rather than a mode of
``nvfp4_cutedsl._quantize_kernel``:

* the block scale carries an MSE correction built from two sequential fp32 FMA
  chains over the block, and is then stochastically rounded; the data is RNE;
* the random word is indexed by kitchen's tile/thread -> Philox *subsequence*
  map, which is a function of the launch geometry of a kernel this one does not
  otherwise resemble, not of a flat element index;
* the decode numerator is ``6 * 256``, and a zero block scale clamps the encode
  scale to ``FLT_MAX`` instead of falling back to unit scaling.

One thread still owns one 16-element block, which is the granularity both the
correction and the scale RNG work at.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .eden_reference import (
    E4M3_EXPONENT_SCALE,
    E4M3_MANTISSA_BITS,
    EDEN_NUMERATOR,
    FP4_E2M1_MAXVAL,
    THREADS_PER_BLOCK,
    THREADS_STORE,
    TILE_DIM,
)
from .nvfp4_cutedsl import (
    _F32,
    _I32,
    _THREADS,
    _U32,
    _U64,
    _apply_sign,
    _cdiv,
    _e4m3_rne,
    _fp4_e2m1_rne,
    _layout_key,
    _philox4_counter,
    _run,
    _stream,
    compute_amax,
)
from .nvfp4_reference import BLOCK_SIZE, NVFP4Quantized, pad_2d

_FLT_MAX = 3.4028234663852886e38


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@cute.jit
def _emulate_sr_e4m3(v: _F32, rbits: _U32) -> _F32:
    """``emulate_sr_e4m3<false>`` -> ``emulate_rs<120, 3>`` (kitchen rounding.cuh).

    Scale down so the E4M3 subnormals are normal fp32, add the low 20 random
    bits into the fp32 bit pattern, truncate them, scale back. Both scale
    factors are exact powers of two.
    """
    throwaway = _U32(23 - E4M3_MANTISSA_BITS)
    mask = _U32((1 << (23 - E4M3_MANTISSA_BITS)) - 1)
    scaled = v * _F32(2.0**-E4M3_EXPONENT_SCALE)
    noisy = scaled.bitcast(_U32) + (rbits & mask)
    noisy = (noisy >> throwaway) << throwaway
    return noisy.bitcast(_F32) * _F32(2.0**E4M3_EXPONENT_SCALE)


@cute.jit
def _eden_rbits(
    seed: _U64, row: _I32, blk: _I32, grid_other: _I32, transpose: cutlass.Constexpr
) -> _U32:
    """The one random word kitchen's kernel rounds this block's scale with.

    ``row`` / ``blk`` index the *logical* view (blocks along the last axis), so
    on the transpose path ``row`` is an original column and ``blk`` a group of
    16 original rows. ``grid_other`` is ``gridDim.x`` on the identity path and
    ``gridDim.y`` on the transpose path -- the kernel's grid is
    ``(ceil(N/128), ceil(M/128))`` with 256 threads, and the transpose path
    swaps the two because the reference kernel it imitates runs on ``x.T``.

    See ``eden_reference.block_subsequence`` for the derivation; this is the
    same map, expressed per thread.
    """
    seq = _I32(0)
    word = _I32(0)
    if transpose:
        bx = row // _I32(TILE_DIM)
        j_local = row % _I32(TILE_DIM)
        it = j_local // _I32(64)
        rem = j_local % _I32(64)
        tid = (rem // _I32(2)) * _I32(THREADS_STORE) + blk % _I32(THREADS_STORE)
        seq = (
            bx * _I32(THREADS_PER_BLOCK) * grid_other
            + (blk // _I32(THREADS_STORE)) * _I32(THREADS_PER_BLOCK)
            + tid
        )
        word = it * _I32(2) + rem % _I32(2)
    else:
        r_local = row % _I32(TILE_DIM)
        tid = (r_local % _I32(32)) * _I32(THREADS_STORE) + blk % _I32(THREADS_STORE)
        seq = (
            (row // _I32(TILE_DIM)) * _I32(THREADS_PER_BLOCK) * grid_other
            + (blk // _I32(THREADS_STORE)) * _I32(THREADS_PER_BLOCK)
            + tid
        )
        word = r_local // _I32(32)

    # Counter is {offset_lo, offset_hi, subsequence_lo, subsequence_hi}; four
    # blocks per thread and a pre-filled generate4() mean offset is always 0.
    s = _U64(seq)
    w0, w1, w2, w3 = _philox4_counter(
        seed,
        _U32(0),
        _U32(0),
        _U32(s & _U64(0xFFFFFFFF)),
        _U32((s >> _U64(32)) & _U64(0xFFFFFFFF)),
    )
    out = w0
    if word == _I32(1):
        out = w1
    elif word == _I32(2):
        out = w2
    elif word == _I32(3):
        out = w3
    return out


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


@cute.kernel
def _eden_quantize_kernel(
    gX: cute.Tensor,  # (R, C) logical view, C a multiple of 16
    gQ: cute.Tensor,  # (R, C) fp32 FP4 code values
    gS: cute.Tensor,  # (R, C // 16) fp32 corrected + rounded block scale
    gAmax: cute.Tensor,  # (1,) fp32, over the unpadded tensor
    gGdesc: cute.Tensor,  # (1,) fp32 global descale
    n_blocks: _I32,
    blocks_per_row: _I32,
    seed: _U64,
    grid_other: _I32,
    use_sr: cutlass.Constexpr,
    transpose: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    blk = _I32(bidx * _THREADS + tidx)
    if blk < n_blocks:
        row = blk // blocks_per_row
        bcol = blk % blocks_per_row
        col0 = bcol * _I32(BLOCK_SIZE)

        block_amax = _F32(0.0)
        for j in cutlass.range_constexpr(BLOCK_SIZE):
            v = _F32(gX[row, col0 + _I32(j)])
            block_amax = cute.arch.fmax(block_amax, cute.math.absf(v))

        # ComputeGlobalEncodeScaleFP4(amax, 1536): clamp to FLT_MAX, then unit
        # scaling if the tensor is all zero or the scale underflowed.
        amax = _F32(gAmax[0])
        global_encode = _F32(EDEN_NUMERATOR) / amax
        if global_encode > _F32(_FLT_MAX):
            global_encode = _F32(_FLT_MAX)
        if (amax == _F32(0.0)) or (global_encode == _F32(0.0)):
            global_encode = _F32(1.0)
        global_decode = _F32(1.0) / global_encode

        # ComputeEdenBlockDecodeScale -> the uncorrected E4M3 block scale.
        scale_inv = (block_amax * _F32(1.0 / FP4_E2M1_MAXVAL)) * global_encode
        if scale_inv > _F32(_FLT_MAX):
            scale_inv = _F32(_FLT_MAX)
        scale_inv = _e4m3_rne(scale_inv)

        # ComputeEncodeScaleFP4<float, false>. Unlike the psx path there is no
        # "unit scale when the block scale is zero" branch: it clamps instead.
        encode_scale = _F32(1.0) / (scale_inv * global_decode)
        if encode_scale > _F32(_FLT_MAX):
            encode_scale = _F32(_FLT_MAX)

        # Two sequential fp32 FMA chains over the block, in natural element
        # order. ``a * b + c`` lowers to fma.rn.f32 here, which is what the
        # kernel's explicit fma() calls emit and what the reference emulates.
        sum_sq = _F32(0.0)
        sum_cross = _F32(0.0)
        for j in cutlass.range_constexpr(BLOCK_SIZE):
            col = col0 + _I32(j)
            v = _F32(gX[row, col]) * encode_scale
            code = _apply_sign(_fp4_e2m1_rne(cute.math.absf(v)), v)
            gQ[row, col] = code
            sum_sq = v * v + sum_sq
            sum_cross = v * code + sum_cross

        correction = _F32(1.0)
        if sum_cross != _F32(0.0):
            ratio = sum_sq / sum_cross
            if (ratio - ratio) == _F32(0.0):  # isfinite: NaN and +-inf both fail
                correction = ratio
        corrected = scale_inv * correction

        if use_sr:
            corrected = _emulate_sr_e4m3(
                corrected, _eden_rbits(seed, row, bcol, grid_other, transpose)
            )
        gS[row, bcol] = _e4m3_rne(corrected)
        if blk == _I32(0):
            gGdesc[0] = global_decode


@cute.jit
def _launch_eden_quantize(
    mX: cute.Tensor,
    mQ: cute.Tensor,
    mS: cute.Tensor,
    mAmax: cute.Tensor,
    mGdesc: cute.Tensor,
    n_blocks: _I32,
    blocks_per_row: _I32,
    seed: _U64,
    grid_other: _I32,
    grid: _I32,
    use_sr: cutlass.Constexpr,
    transpose: cutlass.Constexpr,
    stream,
):
    _eden_quantize_kernel(
        mX, mQ, mS, mAmax, mGdesc, n_blocks, blocks_per_row, seed, grid_other,
        use_sr, transpose,
    ).launch(grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# Host entry points
# ---------------------------------------------------------------------------


def quantize(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    seed: int = 0,
    stochastic_round_scale: bool = True,
) -> NVFP4Quantized:
    """MS-EDEN NVFP4 quantize of a 2D tensor; logical (block-last) layout."""
    assert x.dim() == 2, "only 2D tensors"
    rows, cols = x.shape
    # The Eden op does not pad; only the block axis is rounded up, and the lanes
    # that adds are the ones the kernel clears in shared memory.
    x_padded = pad_2d(x, *((BLOCK_SIZE, 1) if transpose else (1, BLOCK_SIZE)))
    x_padded = x_padded.contiguous()
    amax = compute_amax(x)

    data_q = torch.empty(x_padded.shape, dtype=torch.float32, device=x.device)
    view_x = x_padded.t() if transpose else x_padded
    view_q = data_q.t() if transpose else data_q
    n_rows, n_cols = view_x.shape
    blocks_per_row = n_cols // BLOCK_SIZE
    n_blocks = n_rows * blocks_per_row
    grid_other = _cdiv(rows if transpose else cols, TILE_DIM)

    block_descale = torch.empty(
        (n_rows, blocks_per_row), dtype=torch.float32, device=x.device
    )
    global_descale = torch.empty(1, dtype=torch.float32, device=x.device)
    tensors = (
        from_dlpack(view_x),
        from_dlpack(view_q),
        from_dlpack(block_descale),
        from_dlpack(amax),
        from_dlpack(global_descale),
    )
    scalars = (
        n_blocks,
        blocks_per_row,
        seed,
        grid_other,
        _cdiv(n_blocks, _THREADS),
    )
    key = ("eden", stochastic_round_scale, transpose) + _layout_key(
        view_x, view_q, block_descale
    )
    _run(
        key,
        _launch_eden_quantize,
        tensors + scalars + (stochastic_round_scale, transpose, _stream()),
        tensors + scalars + (_stream(),),
    )
    return NVFP4Quantized(
        data_q=view_q, block_descale=block_descale, global_descale=global_descale
    )


def quant_dequant(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    use_sr: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Full leaf QDQ: bf16 in ``x``'s layout, trimmed back to ``x``'s shape.

    ``use_sr`` names the *scale* rounding, matching ``eden_reference``.
    """
    from .nvfp4_cutedsl import dequantize

    q = quantize(x, transpose=transpose, seed=seed, stochastic_round_scale=use_sr)
    dq = dequantize(q)
    if transpose:
        dq = dq.t()
    return dq[: x.shape[0], : x.shape[1]]
