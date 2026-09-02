"""Self-contained PyTorch reference for kitchen's MS-EDEN NVFP4 QDQ (recipe 100483).

Recipe 100483 quantizes G / G.T with ``QuantizeOpEdenFP4Emulation`` instead of
``QuantizeOpNVFP4Emulation``. The difference from recipe 9004 is not a tweak to
the same algorithm, it is a different one:

* the FP4 **data** is round-to-nearest-even, never stochastic;
* the **E4M3 block scale** carries an MSE correction and is then *stochastically
  rounded* -- the scale is the only random step;
* the per-tensor decode numerator is ``6 * 256 = 1536``, not ``6 * 448 = 2688``.
  The block-scale ceiling is 256 rather than 448 so the correction (which can
  only raise the scale) has headroom before it saturates.

This transcribes
``kitchen/csrc/ops/quantize/quantize_transpose_vector_blockwise_fp4_eden.cu``
for the configuration 100483 uses: ``correction_dim=16``, ``ue5m3_scale=False``,
``stochastic_round_scale=True`` and ``eden_dot_product_type=torch.float32`` (so
the fp32 FMA accumulators, not the bf16x2 / fp16x2 ones), reached through
``QuantizeOpEdenFP4Emulation.quant_dequant_kernel`` ->
``ops.quantize_vector_blockwise_fp4_eden``.

Unlike the psx "clippy" emulation path, the Eden op does **no padding**: it hands
the tensor to the kernel as-is and the kernel masks out-of-range lanes. Rows past
the end are cleared to zero in shared memory, so the model here is "zero-pad the
block axis up to a multiple of 16, quantize, crop".

Bitwise fidelity, and why it is reachable here when it was not for 9004
----------------------------------------------------------------------
Under 9004 the FP4 data is stochastically rounded by ``curand()`` *inside* the
psx kernel, so the random stream can never be shared with an outside
implementation. Under 100483 the only random step is
``emulate_sr_e4m3`` (``kitchen/csrc/rounding.cuh``), which is **software** bit
manipulation on one Philox word whose ``(seed, subsequence, offset)`` layout is
documented, and whose seed is a plain scalar kernel argument. So this module
reproduces kitchen's *actual* stream rather than substituting its own:
:func:`block_subsequence` re-derives the kernel's tile/thread -> subsequence map
and :func:`philox_rbits` re-derives the words.

``kitchen/testing/philox_testing_utils.py`` is kitchen's own NumPy model of the
same two pieces; this is a torch transcription of it.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .nvfp4_reference import (
    BLOCK_SIZE,
    NVFP4Quantized,
    _U32,
    _mulhilo,
    cast_to_fp4_e2m1,
    dequantize,
)

FP4_E2M1_MAXVAL = 6.0
# kEdenBlockScaleMax / kEdenNumerator (the .cu, :33-42). 256, not E4M3's 448.
EDEN_BLOCK_SCALE_MAX = 256.0
EDEN_NUMERATOR = FP4_E2M1_MAXVAL * EDEN_BLOCK_SCALE_MAX  # 1536.0

# emulate_sr_e4m3 instantiates emulate_rs<120, 3> (rounding.cuh:60-72).
E4M3_EXPONENT_SCALE = 120
E4M3_MANTISSA_BITS = 3

# Kernel launch geometry, which the Philox subsequence map is a function of:
# dim3 grid(ceil(N/128), ceil(M/128)) and 256 threads (the .cu, :135, :924, :959).
TILE_DIM = 128
THREADS_PER_BLOCK = 256
# kNumThreadsStore = kTileDim / kNVecOut = 8 on both paths.
THREADS_STORE = 8

FLT_MAX = torch.finfo(torch.float32).max


# ---------------------------------------------------------------------------
# Philox-4x32-10, cuRANDDx counter layout
# ---------------------------------------------------------------------------


def philox_rbits(subsequence: torch.Tensor, seed: int, offset: int = 0) -> torch.Tensor:
    """The four uint32 of ``Philox10RoundRNG(seed, subsequence, offset).generate4()``.

    Counter words are ``{offset_lo, offset_hi, subsequence_lo, subsequence_hi}``
    and key words ``{seed_lo, seed_hi}`` -- kitchen's ``philox4x32_10``. Returns
    ``subsequence.shape + (4,)`` int64 holding values in ``[0, 2**32)``, in
    consumption order.

    This is a *different* parameterisation from
    ``nvfp4_reference.philox_uniform``, which is keyed on a flat element index
    and returns 24-bit floats. Eden needs raw words at an explicit subsequence,
    so the two cannot share code.
    """
    c0 = torch.full_like(subsequence, offset & _U32)
    c1 = torch.full_like(subsequence, (offset >> 32) & _U32)
    c2 = subsequence & _U32
    c3 = (subsequence >> 32) & _U32
    k0 = seed & _U32
    k1 = (seed >> 32) & _U32
    for _ in range(10):
        hi0, lo0 = _mulhilo(0xD2511F53, c0)
        hi1, lo1 = _mulhilo(0xCD9E8D57, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _U32, lo1, (hi0 ^ c3 ^ k1) & _U32, lo0
        k0 = (k0 + 0x9E3779B9) & _U32
        k1 = (k1 + 0xBB67AE85) & _U32
    return torch.stack((c0, c1, c2, c3), dim=-1)


def block_subsequence(
    rows: int,
    cols: int,
    transpose: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-16-element-block ``(subsequence, word_index)`` for the Eden kernel.

    ``rows``/``cols`` are the *original* tensor's shape ``(M, N)``; the returned
    tensors are indexed ``[row, block]`` in the logical layout the caller
    quantizes in (blocks along the last axis of the logical view), so they line
    up elementwise with ``block_amax``.

    Identity path (the .cu, :218 and :284-300). ``blockDim`` is ``(256, 1, 1)``,
    which collapses ``blockDim.y = 1`` and ``threadIdx.y = 0`` out of the
    published expression, leaving::

        seq = blockIdx.y * 256 * gridDim.x + blockIdx.x * 256 + threadIdx.x

    Each thread owns 16 consecutive columns of one row and walks four rows
    ``32`` apart, consuming one word per iteration.

    Transpose path (the .cu, :229 and :560-577), whose thread mapping the kernel
    spells out for exactly this purpose: ``c_s = tx/8``, ``r_s = (tx%8)*16``,
    with the block order per thread being ``iter``-major over ``smem_idx``.
    ``blockIdx.x`` and ``gridDim.y`` swap roles because the reference kernel
    this path imitates runs on ``x.T``.
    """
    grid_x = -(-cols // TILE_DIM)
    grid_y = -(-rows // TILE_DIM)

    if not transpose:
        # Logical view is (M, ceil(N/16)): block b of row r covers columns 16b..16b+16.
        r = torch.arange(rows, device=device, dtype=torch.int64).reshape(-1, 1)
        c = torch.arange(
            -(-cols // BLOCK_SIZE), device=device, dtype=torch.int64
        ).reshape(1, -1)
        by, r_local = r // TILE_DIM, r % TILE_DIM
        bx, c_local = c // THREADS_STORE, c % THREADS_STORE
        r_s, word = r_local % 32, r_local // 32
        tid = r_s * THREADS_STORE + c_local
        seq = by * THREADS_PER_BLOCK * grid_x + bx * THREADS_PER_BLOCK + tid
        return torch.broadcast_tensors(seq, word.expand_as(seq))

    # Transpose: logical view is (N, ceil(M/16)); "row" j is an original column
    # and "block" R16 is a group of 16 original rows.
    j = torch.arange(cols, device=device, dtype=torch.int64).reshape(-1, 1)
    r16 = torch.arange(-(-rows // BLOCK_SIZE), device=device, dtype=torch.int64).reshape(
        1, -1
    )
    bx, j_local = j // TILE_DIM, j % TILE_DIM
    it, rem = j_local // 64, j_local % 64
    c_s, smem_idx = rem // 2, rem % 2
    by, r_s8 = r16 // THREADS_STORE, r16 % THREADS_STORE
    tid = c_s * THREADS_STORE + r_s8
    seq = bx * THREADS_PER_BLOCK * grid_y + by * THREADS_PER_BLOCK + tid
    word = it * 2 + smem_idx
    return torch.broadcast_tensors(seq, word.expand_as(seq))


def block_rbits(
    rows: int, cols: int, transpose: bool, seed: int, device: torch.device
) -> torch.Tensor:
    """The single random word each 16-element block's scale is rounded with."""
    seq, word = block_subsequence(rows, cols, transpose, device)
    # Four blocks per thread and get_rbits() pre-fills from one generate4(), so
    # every word this kernel consumes comes from the offset-0 draw.
    words = philox_rbits(seq, seed)
    return words.gather(-1, word.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Software stochastic rounding (kitchen/csrc/rounding.cuh)
# ---------------------------------------------------------------------------


def emulate_rs(
    values: torch.Tensor,
    rbits: torch.Tensor,
    exponent_scale: int = E4M3_EXPONENT_SCALE,
    mantissa_bits: int = E4M3_MANTISSA_BITS,
) -> torch.Tensor:
    """``emulate_rs<kExponentScale, kOutputMantissaBits>`` (rounding.cuh:26-58).

    Scale by ``2**-exponent_scale`` so the target format's subnormals are normal
    in fp32, add the low ``23 - mantissa_bits`` random bits into the fp32 bit
    pattern, truncate those bits, scale back.
    """
    throwaway = 23 - mantissa_bits
    mask = (1 << throwaway) - 1
    scaled = (values.float() * (2.0**-exponent_scale)).float()
    bits = scaled.view(torch.int32).to(torch.int64) & _U32
    noisy = (bits + (rbits & mask)) & _U32
    noisy = (noisy >> throwaway) << throwaway
    unscaled = noisy.to(torch.int32).view(torch.float32)
    return (unscaled * (2.0**exponent_scale)).float()


def emulate_sr_e4m3(values: torch.Tensor, rbits: torch.Tensor) -> torch.Tensor:
    """``emulate_sr_e4m3<false>`` -- non-saturating; the E4M3 cast saturates."""
    return emulate_rs(values, rbits)


# ---------------------------------------------------------------------------
# Exact fp32 FMA, for the correction dot products
# ---------------------------------------------------------------------------


def _fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """``fmaf(a, b, c)``: ``a * b + c`` with a *single* fp32 rounding.

    The kernel accumulates ``sum_sq`` / ``sum_cross`` with sequential ``fma()``
    calls (the .cu, :415-423), so a multiply-then-add reference is off by an ULP
    on some blocks -- enough to move the corrected scale one E4M3 step. torch
    exposes no fp32 FMA, so compute in float64 (where ``a * b`` is exact) and
    round to odd before narrowing, which makes the double rounding agree with a
    single fp32 rounding of the exact value.
    """
    ad, bd, cd = a.double(), b.double(), c.double()
    product = ad * bd  # exact: 24 + 24 <= 53 bits
    total = product + cd  # correctly rounded to float64
    # Two-sum residual: exact, and nonzero exactly when the add rounded.
    shifted = total - product
    err = (product - (total - shifted)) + (cd - shifted)
    bits = total.view(torch.int64)
    sticky = (err != 0) & ((bits & 1) == 0) & torch.isfinite(total)
    total = torch.where(sticky, bits | 1, bits).view(torch.float64)
    return total.float()


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------


def _logical_view(x: torch.Tensor, transpose: bool) -> torch.Tensor:
    """View whose last dim is the 16-element block axis (identity: rows)."""
    return x.t() if transpose else x


def quantize(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    seed: int = 0,
    stochastic_round_scale: bool = True,
) -> NVFP4Quantized:
    """MS-EDEN NVFP4 quantize of a 2D tensor; returns the logical-layout result.

    ``stochastic_round_scale=False`` is kitchen's deterministic branch (RNE on
    the corrected scale), which exists so a test can isolate the correction from
    the RNG.
    """
    assert x.dim() == 2, "only 2D tensors"
    rows, cols = x.shape

    # ops.compute_tensor_absmax over the *unpadded* tensor: the Eden op does not
    # pad, so there is no padding for the amax to see.
    amax = x.abs().float().amax()

    view = _logical_view(x, transpose)
    n_rows, n_cols = view.shape
    pad = -n_cols % BLOCK_SIZE
    if pad:
        # Lanes past the end are cleared in shared memory, i.e. zero.
        view = torch.nn.functional.pad(view, (0, pad), value=0.0)
    n_blocks = view.shape[1] // BLOCK_SIZE
    blocked = view.reshape(n_rows, n_blocks, BLOCK_SIZE).float()

    # ComputeGlobalEncodeScaleFP4(amax, 1536): clamp to FLT_MAX, then fall back
    # to unit scaling if the amax is zero or the scale underflowed.
    global_encode = torch.div(torch.full_like(amax, EDEN_NUMERATOR), amax)
    global_encode = torch.clamp(global_encode, max=FLT_MAX)
    global_encode = torch.where(
        (amax == 0) | (global_encode == 0), torch.ones_like(global_encode), global_encode
    )
    global_decode = torch.reciprocal(global_encode)

    # ComputeEdenBlockDecodeScale -> ComputeDecodeScaleFP4<..., false>: the
    # uncorrected E4M3 block scale, RNE, saturating.
    block_amax = blocked.abs().amax(dim=-1, keepdim=True)
    scale_inv = (block_amax * (1.0 / FP4_E2M1_MAXVAL)) * global_encode
    scale_inv = torch.clamp(scale_inv, max=FLT_MAX)
    scale_inv = scale_inv.to(torch.float8_e4m3fn).float()

    # ComputeEncodeScaleFP4<float, false>. Note this is *not* the psx path's
    # "unit scale when the block scale is zero" -- Eden clamps to FLT_MAX, and a
    # zero block scale therefore feeds a zero block through to a zero output.
    encode_scale = torch.clamp(
        torch.reciprocal(scale_inv * global_decode), max=FLT_MAX
    )

    scaled = blocked * encode_scale
    codes = cast_to_fp4_e2m1(scaled)

    # Sequential fp32 FMA over the 16 elements, in natural order, two chains.
    sum_sq = torch.zeros_like(scale_inv).squeeze(-1)
    sum_cross = torch.zeros_like(sum_sq)
    for i in range(BLOCK_SIZE):
        xi, ci = scaled[..., i], codes[..., i]
        sum_sq = _fma(xi, xi, sum_sq)
        sum_cross = _fma(xi, ci, sum_cross)
    sum_sq, sum_cross = sum_sq.unsqueeze(-1), sum_cross.unsqueeze(-1)

    # Fall back to 1 when the block quantized to all zeros or the ratio blew up.
    ratio = sum_sq / sum_cross
    correction = torch.where(
        (sum_cross != 0) & torch.isfinite(ratio), ratio, torch.ones_like(ratio)
    )
    corrected = scale_inv * correction

    if stochastic_round_scale:
        rbits = block_rbits(rows, cols, transpose, seed, x.device)
        corrected = emulate_sr_e4m3(corrected, rbits.unsqueeze(-1))
    block_descale = corrected.to(torch.float8_e4m3fn).float()

    return NVFP4Quantized(
        data_q=codes.reshape(n_rows, n_blocks * BLOCK_SIZE),
        block_descale=block_descale.reshape(n_rows, n_blocks),
        global_descale=global_decode.reshape(1),
    )


def quant_dequant(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    use_sr: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Full leaf QDQ: bf16 in ``x``'s layout, trimmed back to ``x``'s shape.

    ``use_sr`` names the *scale* rounding here, not the data rounding, and is
    accepted so this module is drop-in for ``nvfp4_reference.quant_dequant``.
    Recipe 100483 always leaves it on.
    """
    q = quantize(x, transpose=transpose, seed=seed, stochastic_round_scale=use_sr)
    dq = dequantize(q)
    if transpose:
        dq = dq.t()
    return dq[: x.shape[0], : x.shape[1]]
