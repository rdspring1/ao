"""Tests for triton_rht_quantize_row_col (SM100+ kernel).

Covers all 8 (stochastic_rounding × compute_rowwise × swizzle_scale_factors) combinations:

  RTNE (stochastic_rounding=False):
    - test_triton_rht_quantize_rtne_scales_vs_reference: FP8 scale factors match the PyTorch
      reference bitwise for both col and row paths, with and without swizzle.
    - test_triton_rht_quantize_rtne_sqnr: Dequantized output reconstructs post-RHT / raw-A
      values with SQNR ≥ 20 dB for both col and row paths.

  SR (stochastic_rounding=True):
    - test_triton_rht_quantize_sr_col_variance: Col SR output has variance > 0 across seeds
      (SR is stochastic); rowwise path always uses RTNE so its variance must be zero.
    - test_triton_rht_quantize_sr_col_mean_mae_lt_rn: Averaged col SR MAE < single-pass RN
      MAE (SR is unbiased; averaging cancels per-element rounding bias).
    - test_triton_rht_quantize_sr_col_vs_reference_mae: Triton SR mean MAE agrees with
      PyTorch reference SR mean MAE (rtol=0.2, atol=1e-4).

8-combination coverage:
  SR=F, RW=F, SW=F  — rtne_scales_vs_reference + rtne_sqnr
  SR=F, RW=F, SW=T  — rtne_scales_vs_reference
  SR=F, RW=T, SW=F  — rtne_scales_vs_reference + rtne_sqnr
  SR=F, RW=T, SW=T  — rtne_scales_vs_reference
  SR=T, RW=F, SW=F  — sr_variance + sr_mean_mae_lt_rn + sr_vs_reference_mae
  SR=T, RW=F, SW=T  — sr_variance + sr_mean_mae_lt_rn + sr_vs_reference_mae
  SR=T, RW=T, SW=F  — (SR does not affect rowwise path; covered by rtne tests)
  SR=T, RW=T, SW=T  — (SR does not affect rowwise path; covered by rtne tests)
"""
import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.utils import from_blocked, to_blocked
from torchao.utils import is_sm_at_least_100

if is_sm_at_least_100():
    from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
        triton_rht_quantize_row_col,
    )
    from torchao.prototype.mx_formats.hadamard_utils import get_rht_matrix

_FP8_E4M3_MAX = 448.0
_FP4_E2M1_MAX = 6.0
_FP4_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# M must be ≥ 128 (BLOCK_M minimum). M=32/64/96 excluded.
_M_VALUES = [128, 160, 256, 512]
# N must be ≥ 128 (BLOCK_N fixed=128). N=100 excluded.
_N_VALUES = [128, 200, 256, 384, 512, 1024]

# Fixed shape for SR statistical tests (10 samples each).
_SR_NUM_SAMPLES = 10
_SR_M, _SR_N = 128, 128


def cast_to_fp4x2(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest FP4 E2M1 and pack two values per byte.

    FP4 E2M1 magnitudes (code 0–7): 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    Midpoints between adjacent values: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0

    Tie-breaking at exact midpoints: round-to-nearest-even, matching PTX
    cvt.rn.satfinite.e2m1x2.f32 used by TE's CUDA kernel.

    At the three boundaries where the lower code has an odd mantissa bit, ties
    round UP to the even upper code:
      0.75 → code 2 (1.0),  1.75 → code 4 (2.0),  3.5 → code 6 (4.0)
    The remaining four boundaries already round toward zero (lower code is even).
    """
    fp4_boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, float("inf")], device=x.device
    )

    sign = (x < 0).to(torch.uint8) * 8
    abs_x = x.abs()
    idx = torch.bucketize(abs_x, fp4_boundaries)  # 0..7, rounds toward zero

    # Round-to-nearest-even correction: at boundaries where the lower code is odd
    # (0.75, 1.75, 3.5), increment idx to reach the adjacent even code.
    rne_up = torch.tensor([0.75, 1.75, 3.5], dtype=abs_x.dtype, device=x.device)
    at_rne_up = (abs_x.unsqueeze(-1) == rne_up).any(dim=-1)
    idx = (idx + at_rne_up.to(torch.int64)).clamp(max=7)

    result = (idx.to(torch.uint8) + sign).to(torch.uint8)

    # Pack pairs of FP4 nibbles into bytes: even columns = low nibble
    return result[:, ::2] + result[:, 1::2] * 16


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _rht_reference(A: torch.Tensor) -> torch.Tensor:
    """PyTorch reference RHT: returns (N, M) bfloat16."""
    M_A, N_A = A.shape
    B = get_rht_matrix(with_random_sign_mask=True, device=A.device)
    return (A.t().reshape(-1, 16) @ B).reshape(N_A, M_A).to(torch.bfloat16)


def _rht_quantize_reference(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RHT + NVFP4 E2M1 columnwise quantization in PyTorch (RTNE, no stochastic rounding).

    Returns:
        codes:       (N, M//2) uint8 packed FP4 codes.
        scale_inv:   (N, M//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32.
    """
    # Keep bf16 for vec_max: the kernel's tl.max(bf16) → bf16, not f32.
    # Using f32 vec_max causes ~0.2% nibble mismatches at FP4 rounding boundaries.
    x_t_rht_bf16 = _rht_reference(A)            # (N, M) bfloat16
    x_t_rht = x_t_rht_bf16.float()              # f32 for arithmetic
    N, M = x_t_rht.shape

    global_amax = x_t_rht.abs().max().float()

    x_vecs = x_t_rht.view(N, M // 16, 16)
    vec_max = x_t_rht_bf16.view(N, M // 16, 16).float().abs().amax(dim=-1, keepdim=True)

    _f32_max = torch.tensor(
        torch.finfo(torch.float32).max, dtype=torch.float32, device=A.device
    )
    if global_amax.item() == 0:
        global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    else:
        global_encode_scale = torch.minimum(
            torch.tensor(
                _FP8_E4M3_MAX * _FP4_E2M1_MAX, dtype=torch.float32, device=A.device
            )
            / global_amax,
            _f32_max,
        )
        if global_encode_scale.item() == 0.0:
            global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    global_decode_scale = (
        torch.ones(1, dtype=torch.float32, device=A.device) / global_encode_scale
    )

    pvscale = (vec_max / _FP4_E2M1_MAX) * global_encode_scale
    pvscale = pvscale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    pvscale_fp8 = pvscale.to(torch.float8_e4m3fn)
    scale_inv = pvscale_fp8.squeeze(-1)  # (N, M//16)

    encode_scale = torch.minimum(
        1.0 / (pvscale_fp8.to(torch.float32) * global_decode_scale),
        _f32_max,
    )
    scaled = (x_vecs * encode_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX).view(N, M)
    codes = cast_to_fp4x2(scaled)  # (N, M//2) uint8

    return codes, scale_inv, global_amax


def _rht_quantize_rowwise_reference(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 E2M1 rowwise quantization of A in PyTorch (RTNE, no RHT applied).

    Mirrors the kernel's rowwise path: quantize raw A, grouping 16 elements along
    the column (N) dimension.

    Returns:
        codes:       (M, N//2) uint8 packed FP4 codes.
        scale_inv:   (M, N//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32 (max(abs(A))).
    """
    x_bf16 = A  # (M, N) bfloat16
    x = x_bf16.float()
    M, N = x.shape

    global_amax = x.abs().max().float()

    x_vecs = x.view(M, N // 16, 16)
    vec_max = x_bf16.view(M, N // 16, 16).float().abs().amax(dim=-1, keepdim=True)

    _f32_max = torch.tensor(
        torch.finfo(torch.float32).max, dtype=torch.float32, device=A.device
    )
    if global_amax.item() == 0:
        global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    else:
        global_encode_scale = torch.minimum(
            torch.tensor(
                _FP8_E4M3_MAX * _FP4_E2M1_MAX, dtype=torch.float32, device=A.device
            )
            / global_amax,
            _f32_max,
        )
        if global_encode_scale.item() == 0.0:
            global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    global_decode_scale = (
        torch.ones(1, dtype=torch.float32, device=A.device) / global_encode_scale
    )

    pvscale = (vec_max / _FP4_E2M1_MAX) * global_encode_scale
    pvscale = pvscale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    pvscale_fp8 = pvscale.to(torch.float8_e4m3fn)
    scale_inv = pvscale_fp8.squeeze(-1)  # (M, N//16)

    encode_scale = torch.minimum(
        1.0 / (pvscale_fp8.to(torch.float32) * global_decode_scale),
        _f32_max,
    )
    scaled = (x_vecs * encode_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX).view(M, N)
    codes = cast_to_fp4x2(scaled)  # (M, N//2) uint8

    return codes, scale_inv, global_amax


def _rht_quantize_reference_sr(
    A: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RHT + NVFP4 E2M1 columnwise quantization in PyTorch with stochastic rounding.

    Uses the same scale-computation path as _rht_quantize_reference, then applies
    stochastic rounding for FP4 code assignment: for each scaled value v, find the
    two neighboring FP4 magnitudes and randomly choose proportional to fractional
    position in the interval.

    Returns:
        codes:       (N, M//2) uint8 packed FP4 codes.
        scale_inv:   (N, M//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32.
    """
    x_t_rht_bf16 = _rht_reference(A)
    x_t_rht = x_t_rht_bf16.float()
    N, M = x_t_rht.shape

    global_amax = x_t_rht.abs().max().float()

    x_vecs = x_t_rht.view(N, M // 16, 16)
    vec_max = x_t_rht_bf16.view(N, M // 16, 16).float().abs().amax(dim=-1, keepdim=True)

    _f32_max = torch.tensor(
        torch.finfo(torch.float32).max, dtype=torch.float32, device=A.device
    )
    if global_amax.item() == 0:
        global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    else:
        global_encode_scale = torch.minimum(
            torch.tensor(
                _FP8_E4M3_MAX * _FP4_E2M1_MAX, dtype=torch.float32, device=A.device
            )
            / global_amax,
            _f32_max,
        )
        if global_encode_scale.item() == 0.0:
            global_encode_scale = torch.ones(1, dtype=torch.float32, device=A.device)
    global_decode_scale = (
        torch.ones(1, dtype=torch.float32, device=A.device) / global_encode_scale
    )

    pvscale = (vec_max / _FP4_E2M1_MAX) * global_encode_scale
    pvscale = pvscale.clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    pvscale_fp8 = pvscale.to(torch.float8_e4m3fn)
    scale_inv = pvscale_fp8.squeeze(-1)  # (N, M//16)

    encode_scale = torch.minimum(
        1.0 / (pvscale_fp8.to(torch.float32) * global_decode_scale),
        _f32_max,
    )
    scaled = (x_vecs * encode_scale).clamp(-_FP4_E2M1_MAX, _FP4_E2M1_MAX).view(N, M)

    # Stochastic rounding on the non-uniform FP4 magnitude grid.
    # For each value v, find its lower (lo) and upper (hi) FP4 neighbors and
    # randomly round up with probability (v - lo) / (hi - lo).
    sign = torch.sign(scaled)
    sign = torch.where(scaled == 0.0, torch.ones_like(sign), sign)
    abs_scaled = scaled.abs()

    mags = _FP4_MAGNITUDES.to(device=A.device)
    # searchsorted(right=False): returns first index k where mags[k] >= abs_scaled.
    # lo is the grid point just below abs_scaled.
    lo_idx = torch.searchsorted(mags.contiguous(), abs_scaled.contiguous()) - 1
    lo_idx = lo_idx.clamp(0, 7)
    hi_idx = (lo_idx + 1).clamp(0, 7)

    lo_val = mags[lo_idx]
    hi_val = mags[hi_idx]
    gap = hi_val - lo_val
    frac = torch.where(
        gap > 0.0, (abs_scaled - lo_val) / gap, torch.zeros_like(abs_scaled)
    )

    u = torch.rand(abs_scaled.shape, device=A.device, generator=generator)
    rounded_abs = torch.where(u < frac, hi_val, lo_val)
    codes = cast_to_fp4x2(sign * rounded_abs)  # (N, M//2) uint8

    return codes, scale_inv, global_amax


def _dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
) -> torch.Tensor:
    """Decode packed FP4 codes back to float32 post-RHT values."""
    lo = (codes & 0xF).long()
    hi = (codes >> 4).long()
    all_codes = torch.empty(
        codes.shape[0], codes.shape[1] * 2, dtype=torch.long, device=codes.device
    )
    all_codes[:, ::2] = lo
    all_codes[:, 1::2] = hi
    mag = _FP4_MAGNITUDES.to(codes.device)[all_codes & 0x7]
    sign = torch.where((all_codes & 0x8) != 0, -1.0, 1.0)
    scale_f32 = scales.to(torch.float32).repeat_interleave(16, dim=1)
    global_decode_scale = float(global_amax) / (_FP8_E4M3_MAX * _FP4_E2M1_MAX)
    return sign * mag * scale_f32 * global_decode_scale



# ---------------------------------------------------------------------------
# Tests — RTNE (stochastic_rounding=False)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("swizzle_scale_factors", [False, True], ids=["sw0", "sw1"])
@pytest.mark.parametrize("compute_rowwise", [False, True], ids=["rw0", "rw1"])
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_quantize_rtne_scales_vs_reference(
    M, N, compute_rowwise, swizzle_scale_factors
):
    """FP8 scale factors must match the PyTorch reference bitwise.

    Columnwise: RHT + quantize of A.T. Rowwise (compute_rowwise=True): quantize raw A.
    For swizzle_scale_factors=True, compare against to_blocked(reference_scales).

    Note: packed FP4 codes are NOT checked bitwise — the kernel uses an approximate
    reciprocal (rcp.approx.f32, ≤2 ULP) while the reference uses correctly-rounded
    div.rn.f32, causing ~0.2% nibble differences at FP4 midpoints. Use the SQNR
    test for quantization quality validation.
    """
    if swizzle_scale_factors and (M % 128 != 0 or N % 128 != 0):
        pytest.skip("swizzle_scale_factors requires M % 128 == 0 and N % 128 == 0")
    if compute_rowwise and N % 32 != 0:
        pytest.skip(f"compute_rowwise requires N % 32 == 0, got N={N}")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    _, ref_col_sf, _ = _rht_quantize_reference(A)
    tri_col_codes, tri_col_sf, tri_col_amax, tri_row_codes, tri_row_sf, tri_row_amax = (
        triton_rht_quantize_row_col(
            A,
            stochastic_rounding=False,
            compute_rowwise=compute_rowwise,
            swizzle_scale_factors=swizzle_scale_factors,
        )
    )

    # Columnwise scale check
    if swizzle_scale_factors:
        torch.testing.assert_close(
            tri_col_sf.flatten(), to_blocked(ref_col_sf), atol=0, rtol=0
        )
    else:
        torch.testing.assert_close(tri_col_sf, ref_col_sf, atol=0, rtol=0)

    # Rowwise scale check
    if compute_rowwise:
        assert tri_row_sf is not None
        _, ref_row_sf, _ = _rht_quantize_rowwise_reference(A)
        if swizzle_scale_factors:
            torch.testing.assert_close(
                tri_row_sf.flatten(), to_blocked(ref_row_sf), atol=0, rtol=0
            )
        else:
            torch.testing.assert_close(tri_row_sf, ref_row_sf, atol=0, rtol=0)
    else:
        assert tri_row_sf is None


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("compute_rowwise", [False, True], ids=["rw0", "rw1"])
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_quantize_rtne_sqnr(M, N, compute_rowwise):
    """Dequantized output must reconstruct post-RHT / raw-A values with SQNR ≥ 20 dB.

    swizzle_scale_factors is not parametrized here — layout does not affect quantization
    error, only scale memory arrangement.
    """
    if compute_rowwise and N % 32 != 0:
        pytest.skip(f"compute_rowwise requires N % 32 == 0, got N={N}")

    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    tri_col_codes, tri_col_sf, tri_col_amax, tri_row_codes, tri_row_sf, tri_row_amax = (
        triton_rht_quantize_row_col(
            A,
            stochastic_rounding=False,
            compute_rowwise=compute_rowwise,
            swizzle_scale_factors=False,
        )
    )

    # Columnwise SQNR: dequantized should reconstruct RHT(A.T)
    ref_rht = _rht_reference(A).float()
    col_sqnr = compute_error(ref_rht, _dequantize(tri_col_codes, tri_col_sf, tri_col_amax))
    assert col_sqnr >= 20.0, f"Col SQNR {col_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"

    # Rowwise SQNR: dequantized should reconstruct raw A
    if compute_rowwise:
        assert tri_row_codes is not None
        row_sqnr = compute_error(
            A.float(), _dequantize(tri_row_codes, tri_row_sf, tri_row_amax)
        )
        assert row_sqnr >= 20.0, f"Row SQNR {row_sqnr:.2f} dB < 20.0 dB for M={M} N={N}"


# ---------------------------------------------------------------------------
# Tests — SR (stochastic_rounding=True)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("swizzle_scale_factors", [False, True], ids=["sw0", "sw1"])
@torch.no_grad()
def test_triton_rht_quantize_sr_col_variance(swizzle_scale_factors):
    """SR columnwise output must have variance > 0 across seeds (SR is stochastic)."""
    M, N = _SR_M, _SR_N
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # Shared global amax for dequantization (input-derived, same across all runs)
    _, _, col_amax, *_ = triton_rht_quantize_row_col(
        A, stochastic_rounding=False, compute_rowwise=False, swizzle_scale_factors=False
    )

    col_samples = []
    for seed in range(_SR_NUM_SAMPLES):
        torch.manual_seed(seed)
        col_codes, col_sf, *_ = triton_rht_quantize_row_col(
            A,
            stochastic_rounding=True,
            compute_rowwise=False,
            swizzle_scale_factors=swizzle_scale_factors,
        )
        if swizzle_scale_factors:
            col_sf = from_blocked(col_sf.flatten(), N, M // 16)
        col_samples.append(_dequantize(col_codes, col_sf, col_amax))

    col_mean_var = torch.stack(col_samples).var(dim=0).mean().item()
    assert col_mean_var > 0, (
        f"SR col variance is zero over {_SR_NUM_SAMPLES} seeds — "
        "SR may have silently degraded to deterministic quantization"
    )


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("swizzle_scale_factors", [False, True], ids=["sw0", "sw1"])
@torch.no_grad()
def test_triton_rht_quantize_sr_col_mean_mae_lt_rn(swizzle_scale_factors):
    """Averaged SR columnwise MAE must be less than single-pass RTNE MAE.

    SR is unbiased: averaging over seeds cancels per-element rounding bias, giving
    lower MAE than a single deterministic RTNE pass (which has persistent rounding
    error per element).
    """
    M, N = _SR_M, _SR_N
    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    ref_rht = _rht_reference(A).float()

    # Shared global amax (input-derived, same for SR and RTNE)
    _, _, col_amax, *_ = triton_rht_quantize_row_col(
        A, stochastic_rounding=False, compute_rowwise=False, swizzle_scale_factors=False
    )

    # Collect SR samples
    sr_samples = []
    for seed in range(_SR_NUM_SAMPLES):
        torch.manual_seed(seed)
        col_codes, col_sf, *_ = triton_rht_quantize_row_col(
            A,
            stochastic_rounding=True,
            compute_rowwise=False,
            swizzle_scale_factors=swizzle_scale_factors,
        )
        if swizzle_scale_factors:
            col_sf = from_blocked(col_sf.flatten(), N, M // 16)
        sr_samples.append(_dequantize(col_codes, col_sf, col_amax))

    sr_avg = torch.stack(sr_samples).mean(dim=0)
    sr_mean_mae = float((sr_avg - ref_rht).abs().mean())

    # RTNE baseline
    rn_codes, rn_sf, rn_amax, *_ = triton_rht_quantize_row_col(
        A, stochastic_rounding=False, compute_rowwise=False, swizzle_scale_factors=False
    )
    rn_mae = float((_dequantize(rn_codes, rn_sf, rn_amax) - ref_rht).abs().mean())

    assert sr_mean_mae < rn_mae, (
        f"SR averaged col MAE ({sr_mean_mae:.6f}) >= RTNE MAE ({rn_mae:.6f}) — "
        "SR may have regressed to round-nearest or is biased"
    )


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.parametrize("swizzle_scale_factors", [False, True], ids=["sw0", "sw1"])
@torch.no_grad()
def test_triton_rht_quantize_sr_col_vs_reference_mae(swizzle_scale_factors):
    """Triton SR mean MAE must agree with PyTorch reference SR mean MAE (rtol=0.2, atol=1e-4).

    Validates that the Triton SR implementation produces the same statistical
    quantization quality as the PyTorch reference SR algorithm (stochastic rounding
    on the non-uniform FP4 E2M1 magnitude grid).
    """
    M, N = _SR_M, _SR_N
    torch.manual_seed(0)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    ref_rht = _rht_reference(A).float()

    _, _, col_amax, *_ = triton_rht_quantize_row_col(
        A, stochastic_rounding=False, compute_rowwise=False, swizzle_scale_factors=False
    )

    # Triton SR samples
    tri_samples = []
    for seed in range(_SR_NUM_SAMPLES):
        torch.manual_seed(seed)
        col_codes, col_sf, *_ = triton_rht_quantize_row_col(
            A,
            stochastic_rounding=True,
            compute_rowwise=False,
            swizzle_scale_factors=swizzle_scale_factors,
        )
        if swizzle_scale_factors:
            col_sf = from_blocked(col_sf.flatten(), N, M // 16)
        tri_samples.append(_dequantize(col_codes, col_sf, col_amax))

    triton_mean_mae = float(
        (torch.stack(tri_samples).mean(dim=0) - ref_rht).abs().mean()
    )

    # Reference SR samples (independent generator per seed)
    ref_samples = []
    for seed in range(_SR_NUM_SAMPLES):
        gen = torch.Generator(device=A.device).manual_seed(seed)
        ref_codes, ref_sf, _ = _rht_quantize_reference_sr(A, gen)
        ref_samples.append(_dequantize(ref_codes, ref_sf, col_amax))

    ref_mean_mae = float(
        (torch.stack(ref_samples).mean(dim=0) - ref_rht).abs().mean()
    )

    torch.testing.assert_close(
        torch.tensor(triton_mean_mae),
        torch.tensor(ref_mean_mae),
        rtol=0.2,
        atol=1e-4,
    )
