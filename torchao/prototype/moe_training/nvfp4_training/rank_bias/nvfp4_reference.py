"""Self-contained PyTorch reference for the NVFP4 emulation QDQ used by kitchen.

This mirrors, op for op, the leaf quantize/dequantize path that
``kitchen/experimental/tensor_dump_analysis/analyze_rank_bias.py`` drives for
recipe 6302 (``QuantizeRecipe.NVFP4_EMULATION`` with ``use_sr`` on G / G.T):

    kitchen.quantization_psx_fp.QuantizeOpNVFP4Emulation.quantize()
      -> pad -> quant_dequant_kernel (identity) / quant_dequant_transpose_kernel
      -> unpad
    ... .dequantize() -> ``.to(torch.bfloat16)``

The arithmetic follows ``kitchen/nvfp_utils.py`` (``to_nvfp_verbose`` /
``from_nvfp_verbose``) for the E2M1, 1x16-tile, per-tensor two-level-scaling,
E4M3_RNE-block-scale configuration, and ``kitchen/cast_utils.py`` for the FP4
rounding rules (``cast_to_fp4_e2m1`` / ``cast_to_fp4_e2m1_sr``).

One deliberate divergence: stochastic rounding draws its uniforms from an
explicit Philox-4x32-10 counter-based stream keyed by ``(seed, element index)``
instead of ``torch.rand`` on the ambient CUDA generator. That makes a trial
reproducible and lets the CuTe DSL kernels be *bitwise* identical to this
reference; the distribution is unchanged.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import torch

FP4_E2M1_MAXVAL = 6.0  # kitchen.cast_utils.FP4_E2M1_MAXVAL
FP8_E4M3_MAXVAL = 448.0  # kitchen.cast_utils.FP8_E4M3_MAXVAL
BLOCK_SIZE = 16  # quant_tile_shape == (1, 16)

# EmulatedFakeQuantOpBase.quantize(): identity uses the op's padding
# requirements, the transpose path is hard-coded to (128, 64).
PAD_IDENTITY: Tuple[int, int] = (32, 16)
PAD_TRANSPOSE: Tuple[int, int] = (128, 64)


class NVFP4Quantized(NamedTuple):
    """Quantized tensor in the *logical* (block-along-last-dim) layout.

    ``data_q`` holds the FP4 code values in an fp32 container (kitchen's
    ``data_hp_q``), ``block_descale`` is kitchen's ``block_descaling_factor_e4m3``
    decoded to fp32 (the value is always exactly on the E4M3 grid), and
    ``global_descale`` is the scalar ``global_descaling_factor``.
    """

    data_q: torch.Tensor  # (R, C) fp32
    block_descale: torch.Tensor  # (R, C // 16) fp32
    global_descale: torch.Tensor  # (1,) fp32


def pad_2d(x: torch.Tensor, row_divisor: int, col_divisor: int) -> torch.Tensor:
    """Zero-pad rows/cols up to the divisors (EmulatedFakeQuantOpBase._pad_tensor)."""
    rows, cols = x.shape
    pad_rows = -rows % row_divisor
    pad_cols = -cols % col_divisor
    if pad_rows == 0 and pad_cols == 0:
        return x
    return torch.nn.functional.pad(x, (0, pad_cols, 0, pad_rows), value=0.0)


# ---------------------------------------------------------------------------
# Philox-4x32-10 uniforms
# ---------------------------------------------------------------------------

PHILOX_M0 = 0xD2511F53
PHILOX_M1 = 0xCD9E8D57
PHILOX_W0 = 0x9E3779B9
PHILOX_W1 = 0xBB67AE85
_U32 = 0xFFFFFFFF


def _mulhilo(a: int, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """32x32 -> (hi, lo) using 16-bit halves so int64 never overflows."""
    a_hi, a_lo = a >> 16, a & 0xFFFF
    b_hi, b_lo = b >> 16, b & 0xFFFF
    p0 = a_lo * b_lo
    mid = a_lo * b_hi + a_hi * b_lo + (p0 >> 16)
    lo = ((mid & 0xFFFF) << 16) | (p0 & 0xFFFF)
    hi = a_hi * b_hi + (mid >> 16)
    return hi, lo


def philox_uniform(numel: int, seed: int, device: torch.device) -> torch.Tensor:
    """``numel`` uniforms in [0, 1), element ``i`` = word ``i % 4`` of counter ``i // 4``.

    Bit-for-bit reproducible and identical to the CuTe DSL device implementation.
    """
    n_ctr = (numel + 3) // 4
    idx = torch.arange(n_ctr, device=device, dtype=torch.int64)
    c0 = idx & _U32
    c1 = (idx >> 32) & _U32
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = seed & _U32
    k1 = (seed >> 32) & _U32
    for _ in range(10):
        hi0, lo0 = _mulhilo(PHILOX_M0, c0)
        hi1, lo1 = _mulhilo(PHILOX_M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & _U32, lo1, (hi0 ^ c3 ^ k1) & _U32, lo0
        k0 = (k0 + PHILOX_W0) & _U32
        k1 = (k1 + PHILOX_W1) & _U32
    words = torch.stack((c0, c1, c2, c3), dim=-1).reshape(-1)[:numel]
    # 24-bit uniform, same construction torch uses for float rand.
    return (words >> 8).to(torch.float32) * (2.0**-24)


# ---------------------------------------------------------------------------
# FP4 E2M1 rounding (kitchen.cast_utils)
# ---------------------------------------------------------------------------


def cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """RNE cast onto {0, .5, 1, 1.5, 2, 3, 4, 6} (cast_utils.cast_to_fp4_e2m1)."""
    sign = torch.sign(x)
    a = torch.abs(x)
    mag = torch.where(
        a <= 0.25,
        0.0,
        torch.where(
            a < 0.75,
            0.5,
            torch.where(
                a <= 1.25,
                1.0,
                torch.where(
                    a < 1.75,
                    1.5,
                    torch.where(
                        a <= 2.5,
                        2.0,
                        torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return mag * sign


def cast_to_fp4_e2m1_sr(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Stochastic cast (cast_utils.cast_to_fp4_e2m1_sr) with explicit uniforms."""
    sign = torch.sign(x)
    a = torch.abs(x)
    high = torch.where(
        a > 4,
        6.0,
        torch.where(
            a > 3,
            4.0,
            torch.where(
                a > 2,
                3.0,
                torch.where(
                    a > 1.5,
                    2.0,
                    torch.where(a > 1.0, 1.5, torch.where(a > 0.5, 1.0, 0.5)),
                ),
            ),
        ),
    )
    low = torch.where(
        a > 4,
        4.0,
        torch.where(
            a > 3,
            3.0,
            torch.where(
                a > 2,
                2.0,
                torch.where(
                    a > 1.5,
                    1.5,
                    torch.where(a > 1.0, 1.0, torch.where(a > 0.5, 0.5, 0.0)),
                ),
            ),
        ),
    )
    prob_up = (a - low) / (high - low)
    return torch.where(u < prob_up, high, low) * sign


# ---------------------------------------------------------------------------
# Quantize / dequantize
# ---------------------------------------------------------------------------


def _logical_view(x_padded: torch.Tensor, transpose: bool) -> torch.Tensor:
    """View whose last dim is the 16-element quantization-block axis.

    Identity path: 1x16 blocks along the last dim. Transpose path: the psx
    transpose kernel blocks 16 elements down each column, which is the identity
    path applied to ``x.T``.
    """
    return x_padded.t() if transpose else x_padded


def quantize(
    x: torch.Tensor,
    *,
    transpose: bool = False,
    use_sr: bool = False,
    seed: int = 0,
) -> NVFP4Quantized:
    """NVFP4 quantize a padded 2D tensor; returns the logical-layout result."""
    assert x.dim() == 2, "only 2D tensors"
    x_padded = pad_2d(x, *(PAD_TRANSPOSE if transpose else PAD_IDENTITY))
    view = _logical_view(x_padded, transpose)
    rows, cols = view.shape
    assert cols % BLOCK_SIZE == 0

    amax = x_padded.abs().float().amax()  # ops.compute_tensor_absmax
    blocked = view.reshape(rows, cols // BLOCK_SIZE, BLOCK_SIZE).float()
    block_amax = blocked.abs().amax(dim=-1, keepdim=True)

    # kitchen writes ``torch.div(6 * 448, global_amax)``. Two-argument torch.div
    # with a Python scalar *numerator* wraps the scalar in a tensor and does a
    # true IEEE division, whereas the ``scalar / tensor`` operator lowers to
    # reciprocal-then-multiply -- 1 ULP apart on a quarter of all inputs. Spell
    # the true division out so the CuTe kernel has an unambiguous div.rn to
    # match and so this does not silently change with the torch version.
    global_scale = torch.div(
        torch.full_like(amax, FP4_E2M1_MAXVAL * FP8_E4M3_MAXVAL), amax
    )
    # _compute_global_scales(): an amax small enough to overflow the scale (or a
    # zero tensor) falls back to unit scaling.
    global_scale = torch.where(
        torch.isinf(global_scale), torch.ones_like(global_scale), global_scale
    )
    global_descale = torch.reciprocal(global_scale)

    # kitchen writes ``torch.div(max_abs_per_block, 6)``. With a Python scalar
    # *denominator* torch.div goes the other way and multiplies by the fp32
    # reciprocal of the scalar, so this one is a multiply.
    block_descale = (block_amax * (1.0 / FP4_E2M1_MAXVAL)) * global_scale
    block_descale = block_descale.to(torch.float8_e4m3fn).float()
    block_scale = torch.where(
        block_descale == 0,
        torch.ones_like(block_descale),
        torch.reciprocal(block_descale * global_descale),
    )

    scaled = blocked * block_scale
    if use_sr:
        u = philox_uniform(scaled.numel(), seed, x.device).reshape(scaled.shape)
        data_q = cast_to_fp4_e2m1_sr(scaled, u)
    else:
        data_q = cast_to_fp4_e2m1(scaled)

    return NVFP4Quantized(
        data_q=data_q.reshape(rows, cols),
        block_descale=block_descale.reshape(rows, cols // BLOCK_SIZE),
        global_descale=global_descale.reshape(1),
    )


def dequantize(q: NVFP4Quantized) -> torch.Tensor:
    """from_nvfp_verbose + QuantizeOpBase.dequantize (cast to bf16), logical layout."""
    block_descale = q.block_descale.repeat_interleave(BLOCK_SIZE, dim=1)
    restored = q.data_q * block_descale * q.global_descale
    return restored.to(torch.bfloat16)


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
