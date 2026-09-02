"""CuTe DSL (nvidia-cutlass-dsl) NVFP4 quantize / dequantize kernels.

Bitwise identical to :mod:`nvfp4_reference`, which in turn mirrors kitchen's
``QuantizeOpNVFP4Emulation`` leaf path (E2M1 data, 1x16 tiles, per-tensor
two-level scaling, E4M3 RNE block scales, RNE or stochastic FP4 rounding).

Three kernels:
  * ``_amax_kernel``      -- per-tensor absmax (kitchen ``compute_tensor_absmax``)
  * ``_quantize_kernel``  -- one thread per 16-element quantization block
  * ``_dequantize_kernel``-- elementwise rescale + bf16 cast

The transpose path (kitchen ``quant_dequant_transpose_kernel``, which blocks 16
elements *down each column*) reuses the same kernels by passing a transposed
CuTe view of the same memory.
"""

from __future__ import annotations

from typing import Tuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .nvfp4_reference import (
    BLOCK_SIZE,
    FP4_E2M1_MAXVAL,
    FP8_E4M3_MAXVAL,
    PAD_IDENTITY,
    PAD_TRANSPOSE,
    PHILOX_M0,
    PHILOX_M1,
    PHILOX_W0,
    PHILOX_W1,
    NVFP4Quantized,
    pad_2d,
)

_THREADS = 128
_F32 = cutlass.Float32
_U32 = cutlass.Uint32
_U64 = cutlass.Uint64
_I32 = cutlass.Int32

_SIGN_BIT = 0x80000000


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


@cute.jit
def _e4m3_rne(v: _F32) -> _F32:
    """Round a non-negative fp32 onto the E4M3 (float8_e4m3fn) grid, RNE + satfinite.

    Matches ``tensor.to(torch.float8_e4m3fn).float()`` on CUDA, which saturates
    (449 -> 448, inf -> 448) rather than producing NaN. Verified exhaustively
    over every fp32 bit pattern in [0, 449) by the test suite.
    """
    res = _F32(448.0)
    if v < _F32(448.0):
        if v < _F32(0.015625):
            # Subnormal E4M3: the grid is the multiples of 2**-9, anchored at 0,
            # so mantissa truncation does not apply. Adding 2**14 (whose fp32 ulp
            # is 2**-9) and subtracting it back is an exact RNE to that grid.
            res = (v + _F32(16384.0)) - _F32(16384.0)
        else:
            # Normal E4M3: keep 3 mantissa bits, RNE. Carrying into the exponent
            # lands on the next binade's first grid point, which is correct.
            u = v.bitcast(_U32)
            rem = u & _U32(0xFFFFF)
            half = _U32(0x80000)
            odd = (u >> _U32(20)) & _U32(1)
            out = u - rem
            if (rem > half) or ((rem == half) and (odd == _U32(1))):
                out = out + _U32(0x100000)
            res = out.bitcast(_F32)
    return res


@cute.jit
def _philox4_counter(
    seed: _U64, c0: _U32, c1: _U32, c2: _U32, c3: _U32
) -> Tuple[_U32, _U32, _U32, _U32]:
    """Philox-4x32-10 bijection over an explicit counter, keyed by ``seed``.

    The two callers disagree only on how the counter is filled: this module's
    FP4 stochastic rounding keys on a flat element index (words 0/1), while
    kitchen's Eden kernel keys on ``(offset, subsequence)`` (words 0/1 and 2/3 --
    see ``eden_cutedsl``). Sharing the round function keeps them from drifting.
    """
    k0 = _U32(seed & _U64(0xFFFFFFFF))
    k1 = _U32((seed >> _U64(32)) & _U64(0xFFFFFFFF))
    for _ in cutlass.range_constexpr(10):
        p0 = _U64(PHILOX_M0) * _U64(c0)
        p1 = _U64(PHILOX_M1) * _U64(c2)
        hi0 = _U32(p0 >> _U64(32))
        lo0 = _U32(p0 & _U64(0xFFFFFFFF))
        hi1 = _U32(p1 >> _U64(32))
        lo1 = _U32(p1 & _U64(0xFFFFFFFF))
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0), lo1, (hi0 ^ c3 ^ k1), lo0
        k0 = k0 + _U32(PHILOX_W0)
        k1 = k1 + _U32(PHILOX_W1)
    return c0, c1, c2, c3


@cute.jit
def _philox4(seed: _U64, ctr: _U64) -> Tuple[_U32, _U32, _U32, _U32]:
    """Philox-4x32-10 with counter ``(ctr_lo, ctr_hi, 0, 0)`` and key ``seed``."""
    return _philox4_counter(
        seed,
        _U32(ctr & _U64(0xFFFFFFFF)),
        _U32((ctr >> _U64(32)) & _U64(0xFFFFFFFF)),
        _U32(0),
        _U32(0),
    )


@cute.jit
def _uniform(word: _U32) -> _F32:
    """24-bit uniform in [0, 1) from a random word."""
    return _F32(word >> _U32(8)) * _F32(2.0**-24)


@cute.jit
def _fp4_e2m1_rne(mag: _F32) -> _F32:
    """RNE onto {0, .5, 1, 1.5, 2, 3, 4, 6} (kitchen cast_to_fp4_e2m1)."""
    out = _F32(6.0)
    if mag <= _F32(0.25):
        out = _F32(0.0)
    elif mag < _F32(0.75):
        out = _F32(0.5)
    elif mag <= _F32(1.25):
        out = _F32(1.0)
    elif mag < _F32(1.75):
        out = _F32(1.5)
    elif mag <= _F32(2.5):
        out = _F32(2.0)
    elif mag < _F32(3.5):
        out = _F32(3.0)
    elif mag <= _F32(5.0):
        out = _F32(4.0)
    return out


@cute.jit
def _fp4_e2m1_sr(mag: _F32, u: _F32) -> _F32:
    """Stochastic rounding onto the E2M1 grid (kitchen cast_to_fp4_e2m1_sr)."""
    high = _F32(0.5)
    low = _F32(0.0)
    if mag > _F32(4.0):
        high, low = _F32(6.0), _F32(4.0)
    elif mag > _F32(3.0):
        high, low = _F32(4.0), _F32(3.0)
    elif mag > _F32(2.0):
        high, low = _F32(3.0), _F32(2.0)
    elif mag > _F32(1.5):
        high, low = _F32(2.0), _F32(1.5)
    elif mag > _F32(1.0):
        high, low = _F32(1.5), _F32(1.0)
    elif mag > _F32(0.5):
        high, low = _F32(1.0), _F32(0.5)
    prob_up = (mag - low) / (high - low)
    out = low
    if u < prob_up:
        out = high
    return out


@cute.jit
def _apply_sign(mag: _F32, src: _F32) -> _F32:
    """``mag * torch.sign(src)``.

    ``torch.sign`` maps both ``+0.0`` and ``-0.0`` to ``+0.0``, so an exactly
    zero input yields ``+0.0`` while a nonzero input that rounds to zero keeps
    its sign (``-0.0``). Carrying the sign bit only for nonzero inputs
    reproduces that, including the signed zeros.
    """
    bits = mag.bitcast(_U32)
    if src != _F32(0.0):
        bits = bits | (src.bitcast(_U32) & _U32(_SIGN_BIT))
    return bits.bitcast(_F32)


@cute.jit
def _sr_store(
    gX: cute.Tensor,
    gQ: cute.Tensor,
    row: _I32,
    col: _I32,
    block_scale: _F32,
    word: _U32,
):
    v = _F32(gX[row, col]) * block_scale
    gQ[row, col] = _apply_sign(_fp4_e2m1_sr(cute.math.absf(v), _uniform(word)), v)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@cute.kernel
def _amax_kernel(gX: cute.Tensor, gAmax: cute.Tensor, numel: _I32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    gdim, _, _ = cute.arch.grid_dim()
    local = _F32(0.0)
    i = _I32(bidx * _THREADS + tidx)
    stride = _I32(gdim * _THREADS)
    while i < numel:
        v = _F32(gX[i])
        local = cute.arch.fmax(local, cute.math.absf(v))
        i += stride
    local = cute.arch.warp_reduction_max(local)
    lane = cute.arch.lane_idx()
    if lane == 0:
        cute.arch.atomic_fmax(gAmax.iterator, local)


@cute.kernel
def _quantize_kernel(
    gX: cute.Tensor,  # (R, C) logical view of the padded input
    gQ: cute.Tensor,  # (R, C) fp32 FP4 code values
    gS: cute.Tensor,  # (R, C // 16) fp32 block descale
    gAmax: cute.Tensor,  # (1,) fp32
    gGdesc: cute.Tensor,  # (1,) fp32 global descale
    n_blocks: _I32,
    blocks_per_row: _I32,
    seed: _U64,
    use_sr: cutlass.Constexpr,
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

        # A true divide here and a reciprocal-multiply below: that is how
        # PyTorch lowers kitchen's two torch.div calls -- see
        # nvfp4_reference.quantize.
        amax = _F32(gAmax[0])
        global_scale = _F32(FP4_E2M1_MAXVAL * FP8_E4M3_MAXVAL) / amax
        if global_scale == _F32(float("inf")):
            global_scale = _F32(1.0)
        global_descale = _F32(1.0) / global_scale

        block_descale = _e4m3_rne(
            (block_amax * _F32(1.0 / FP4_E2M1_MAXVAL)) * global_scale
        )
        block_scale = _F32(1.0)
        if block_descale != _F32(0.0):
            block_scale = _F32(1.0) / (block_descale * global_descale)

        if use_sr:
            base = _U64(row) * _U64(gX.shape[1]) + _U64(col0)
            for g in cutlass.range_constexpr(BLOCK_SIZE // 4):
                w0, w1, w2, w3 = _philox4(seed, (base >> _U64(2)) + _U64(g))
                _sr_store(gX, gQ, row, col0 + _I32(4 * g + 0), block_scale, w0)
                _sr_store(gX, gQ, row, col0 + _I32(4 * g + 1), block_scale, w1)
                _sr_store(gX, gQ, row, col0 + _I32(4 * g + 2), block_scale, w2)
                _sr_store(gX, gQ, row, col0 + _I32(4 * g + 3), block_scale, w3)
        else:
            for j in cutlass.range_constexpr(BLOCK_SIZE):
                col = col0 + _I32(j)
                v = _F32(gX[row, col]) * block_scale
                gQ[row, col] = _apply_sign(_fp4_e2m1_rne(cute.math.absf(v)), v)

        gS[row, bcol] = block_descale
        if blk == _I32(0):
            gGdesc[0] = global_descale


@cute.kernel
def _dequantize_kernel(
    gQ: cute.Tensor,  # (R, C) fp32
    gS: cute.Tensor,  # (R, C // 16) fp32
    gGdesc: cute.Tensor,  # (1,) fp32
    gOut: cute.Tensor,  # (R, C) bf16
    numel: _I32,
    cols: _I32,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = _I32(bidx * _THREADS + tidx)
    if i < numel:
        row = i // cols
        col = i % cols
        v = _F32(gQ[row, col]) * _F32(gS[row, col // _I32(BLOCK_SIZE)])
        gOut[row, col] = cutlass.BFloat16(v * _F32(gGdesc[0]))


# ---------------------------------------------------------------------------
# Host entry points
# ---------------------------------------------------------------------------


@cute.jit
def _launch_amax(mX: cute.Tensor, mAmax: cute.Tensor, numel: _I32, grid: _I32, stream):
    _amax_kernel(mX, mAmax, numel).launch(
        grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )


@cute.jit
def _launch_quantize(
    mX: cute.Tensor,
    mQ: cute.Tensor,
    mS: cute.Tensor,
    mAmax: cute.Tensor,
    mGdesc: cute.Tensor,
    n_blocks: _I32,
    blocks_per_row: _I32,
    seed: _U64,
    grid: _I32,
    use_sr: cutlass.Constexpr,
    stream,
):
    _quantize_kernel(
        mX, mQ, mS, mAmax, mGdesc, n_blocks, blocks_per_row, seed, use_sr
    ).launch(grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream)


@cute.jit
def _launch_dequantize(
    mQ: cute.Tensor,
    mS: cute.Tensor,
    mGdesc: cute.Tensor,
    mOut: cute.Tensor,
    numel: _I32,
    cols: _I32,
    grid: _I32,
    stream,
):
    _dequantize_kernel(mQ, mS, mGdesc, mOut, numel, cols).launch(
        grid=[grid, 1, 1], block=[_THREADS, 1, 1], stream=stream
    )


def _stream() -> cuda_driver.CUstream:
    return cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _layout_key(*tensors: torch.Tensor) -> tuple:
    """Everything about the arguments that the DSL bakes into generated code."""
    return tuple((str(t.dtype), tuple(t.shape), tuple(t.stride())) for t in tensors)


_COMPILED: dict = {}


def _run(key, fn, compile_args, call_args=None):
    """``cute.compile`` once per argument layout; re-tracing per call costs ~40x.

    ``cute.Constexpr`` arguments are baked into the compiled program and must be
    dropped from the invocation, hence the separate ``call_args``.
    """
    entry = _COMPILED.get(key)
    if entry is None:
        entry = cute.compile(fn, *compile_args)
        _COMPILED[key] = entry
    entry(*(compile_args if call_args is None else call_args))


def compute_amax(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor absmax in fp32 (kitchen ``ops.compute_tensor_absmax``)."""
    flat = x.reshape(-1)
    amax = torch.zeros(1, dtype=torch.float32, device=x.device)
    grid = min(_cdiv(flat.numel(), _THREADS), 1024)
    args = (
        from_dlpack(flat),
        from_dlpack(amax),
        flat.numel(),
        grid,
        _stream(),
    )
    _run(("amax",) + _layout_key(flat), _launch_amax, args)
    return amax


def quantize(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    use_sr: bool = False,
    seed: int = 0,
) -> NVFP4Quantized:
    """NVFP4 quantize a 2D tensor; result is in the logical (block-last) layout."""
    assert x.dim() == 2, "only 2D tensors"
    x_padded = pad_2d(x, *(PAD_TRANSPOSE if transpose else PAD_IDENTITY)).contiguous()
    amax = compute_amax(x_padded)

    data_q = torch.empty(x_padded.shape, dtype=torch.float32, device=x.device)
    view_x = x_padded.t() if transpose else x_padded
    view_q = data_q.t() if transpose else data_q
    rows, cols = view_x.shape
    blocks_per_row = cols // BLOCK_SIZE
    n_blocks = rows * blocks_per_row

    block_descale = torch.empty(
        (rows, blocks_per_row), dtype=torch.float32, device=x.device
    )
    global_descale = torch.empty(1, dtype=torch.float32, device=x.device)
    tensors = (
        from_dlpack(view_x),
        from_dlpack(view_q),
        from_dlpack(block_descale),
        from_dlpack(amax),
        from_dlpack(global_descale),
    )
    scalars = (n_blocks, blocks_per_row, seed, _cdiv(n_blocks, _THREADS))
    key = ("quantize", use_sr) + _layout_key(view_x, view_q, block_descale)
    _run(
        key,
        _launch_quantize,
        tensors + scalars + (use_sr, _stream()),
        tensors + scalars + (_stream(),),
    )
    return NVFP4Quantized(
        data_q=view_q, block_descale=block_descale, global_descale=global_descale
    )


def dequantize(q: NVFP4Quantized) -> torch.Tensor:
    """Rescale to bf16 in the logical layout (kitchen ``from_nvfp`` + ``dequantize``)."""
    rows, cols = q.data_q.shape
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=q.data_q.device)
    args = (
        from_dlpack(q.data_q),
        from_dlpack(q.block_descale),
        from_dlpack(q.global_descale),
        from_dlpack(out),
        rows * cols,
        cols,
        _cdiv(rows * cols, _THREADS),
        _stream(),
    )
    key = ("dequantize",) + _layout_key(q.data_q, q.block_descale, out)
    _run(key, _launch_dequantize, args)
    return out


def quant_dequant(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    use_sr: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Full leaf QDQ: bf16 result in ``x``'s layout, trimmed back to ``x``'s shape."""
    q = quantize(x, transpose=transpose, use_sr=use_sr, seed=seed)
    dq = dequantize(q)
    if transpose:
        dq = dq.t()
    return dq[: x.shape[0], : x.shape[1]]
