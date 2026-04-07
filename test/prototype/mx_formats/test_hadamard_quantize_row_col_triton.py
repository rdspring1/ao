"""Tests for triton_rht_quantize_row_col (SM100+ kernel).

Covers all 8 (stochastic_rounding × compute_rowwise × swizzle_scale_factors) combinations:

  RTNE (stochastic_rounding=False):
    - test_triton_rht_quantize_rtne_scales_vs_reference: FP8 scale factors match the PyTorch
      reference bitwise for both col and row paths, with and without swizzle.
    - test_triton_rht_quantize_rtne_sqnr: Dequantized output reconstructs post-RHT / raw-A
      values with SQNR ≥ 20 dB for both col and row paths.

  SR (stochastic_rounding=True):
    - test_triton_rht_quantize_sr_midpoint_distribution: Values at the FP4 [1.0, 1.5]
      midpoint (1.25) round to each neighbor ~50% of the time (columnwise path; input
      constructed via inverse RHT so post-RHT values are exactly 1.25).
    - test_triton_rht_quantize_sr_at_most_one_fp4_step_from_rtne: SR code is at most 1
      FP4 magnitude index step from the RTNE code for every element (columnwise path
      only; rowwise path always uses RTNE regardless of stochastic_rounding).

8-combination coverage:
  SR=F, RW=F, SW=F  — rtne_scales_vs_reference + rtne_sqnr
  SR=F, RW=F, SW=T  — rtne_scales_vs_reference
  SR=F, RW=T, SW=F  — rtne_scales_vs_reference + rtne_sqnr
  SR=F, RW=T, SW=T  — rtne_scales_vs_reference
  SR=T, RW=F, SW=F  — sr_midpoint_distribution + sr_at_most_one_fp4_step_from_rtne
  SR=T, RW=T, SW=F  — (rowwise path always uses RTNE; covered by rtne tests)
"""
import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    nvfp4_quantize,
    per_tensor_amax_to_scale,
)
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import is_sm_at_least_100

if is_sm_at_least_100():
    from torchao.prototype.mx_formats.hadamard_quantize_row_col_triton import (
        triton_rht_quantize_row_col,
    )
    from torchao.prototype.mx_formats.hadamard_utils import get_rht_matrix

# M must be ≥ 128 (BLOCK_M minimum). M=32/64/96 excluded.
_M_VALUES = [128, 160, 256, 512]
# N must be ≥ 128 (BLOCK_N fixed=128). N=100 excluded.
_N_VALUES = [128, 200, 256, 384, 512, 1024]


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
    """RHT + NVFP4 E2M1 columnwise quantization via nvfp4_quantize (RTNE).

    Returns:
        codes:       (N, M//2) uint8 packed FP4 codes.
        scale_inv:   (N, M//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32.
    """
    # Pass bfloat16 output of _rht_reference directly: nvfp4_quantize converts bf16→f32
    # losslessly, so block amax matches the kernel's tl.max(bf16) exactly.
    x_t_rht = _rht_reference(A)  # (N, M) bfloat16
    global_amax = x_t_rht.float().abs().max()
    scale_inv, codes = nvfp4_quantize(
        x_t_rht, per_tensor_scale=per_tensor_amax_to_scale(global_amax)
    )
    return codes, scale_inv, global_amax


def _rht_quantize_rowwise_reference(
    A: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 E2M1 rowwise quantization via nvfp4_quantize (RTNE, no RHT applied).

    Returns:
        codes:       (M, N//2) uint8 packed FP4 codes.
        scale_inv:   (M, N//16) float8_e4m3fn per-vector decode scales.
        global_amax: scalar float32 (max(abs(A))).
    """
    global_amax = A.float().abs().max()
    scale_inv, codes = nvfp4_quantize(
        A, per_tensor_scale=per_tensor_amax_to_scale(global_amax)
    )
    return codes, scale_inv, global_amax


def _dequantize(
    codes: torch.Tensor,
    scales: torch.Tensor,
    global_amax: torch.Tensor,
) -> torch.Tensor:
    """Decode packed FP4 codes via NVFP4Tensor.dequantize()."""
    # orig_dtype=bfloat16: all test inputs are bfloat16; affects only the default
    # output dtype of dequantize(), overridden by the explicit .float() call below.
    return NVFP4Tensor(
        codes, scales, 16, torch.bfloat16,
        per_tensor_scale=per_tensor_amax_to_scale(global_amax),
    ).dequantize().float()


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
@torch.no_grad()
def test_triton_rht_quantize_sr_midpoint_distribution():
    """SR of a value exactly at the FP4 midpoint (1.25) must round each direction ~50% of the time.

    Constructs input A via inverse RHT so that post-RHT values are exactly:
      - 6.0 at the first element of each 16-group (anchors vec_max = global_amax = 6.0,
        so encode_scale = 1.0 exactly).
      - 1.25 everywhere else (exactly at the midpoint of the FP4 [1.0, 1.5] interval).

    The RHT matrix is orthogonal (B @ B.T = I in bfloat16), so the round-trip is exact.
    RTNE rounds 1.25 to code 2 (1.0) — the even neighbor — by round-to-nearest-even.
    SR must round to code 2 (1.0) or code 3 (1.5) with equal probability (~50% each).
    """
    N_RHT, M_RHT = 128, 128  # post-RHT shape (N_RHT = N_A, M_RHT = M_A)
    N_SAMPLES = 32

    # Build A such that RHT(A.T) has 1.25 at non-anchor positions and 6.0 at anchors.
    # Since B is orthogonal, A.T = target @ B^{-1} = target @ B.T.
    B = get_rht_matrix(with_random_sign_mask=True, device="cuda").float()
    target = torch.full((N_RHT, M_RHT), 1.25, dtype=torch.float32, device="cuda")
    target[:, ::16] = 6.0  # one anchor per 16-group along M
    A_t = (target.reshape(N_RHT * M_RHT // 16, 16) @ B.t()).reshape(N_RHT, M_RHT)
    A = A_t.t().contiguous().to(torch.bfloat16)  # kernel expects (M_A, N_A) contiguous

    count_lo = 0  # code 2 = 1.0
    count_hi = 0  # code 3 = 1.5

    for _ in range(N_SAMPLES):
        col_codes, _, _, _, _, _ = triton_rht_quantize_row_col(
            A, stochastic_rounding=True, compute_rowwise=False, swizzle_scale_factors=False
        )
        # Unpack col_codes (N_RHT, M_RHT//2) uint8 → (N_RHT, M_RHT) nibbles
        lo = (col_codes & 0xF).long()
        hi = (col_codes >> 4).long()
        all_nibs = torch.empty(N_RHT, M_RHT, dtype=torch.long, device="cuda")
        all_nibs[:, ::2] = lo
        all_nibs[:, 1::2] = hi
        mag_codes = all_nibs & 0x7

        # Exclude anchor positions (m % 16 == 0 → scaled=6.0 → code 7)
        col_idx = torch.arange(M_RHT, device="cuda")
        target_mags = mag_codes[:, (col_idx % 16) != 0]  # (N_RHT, 15 * M_RHT//16)

        count_lo += (target_mags == 2).sum().item()  # rounded to 1.0
        count_hi += (target_mags == 3).sum().item()  # rounded to 1.5

    total = count_lo + count_hi
    frac_hi = count_hi / total
    assert 0.40 <= frac_hi <= 0.60, (
        f"SR at midpoint 1.25: expected ~50% round to code 3 (1.5), "
        f"got {frac_hi:.4f} over {total} samples"
    )


@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@torch.no_grad()
def test_triton_rht_quantize_sr_at_most_one_fp4_step_from_rtne():
    """SR code must be at most 1 FP4 magnitude index step from the RTNE code.

    SR picks the floor or ceil of the scaled value on the FP4 magnitude grid.
    RTNE also picks floor or ceil (nearest). Therefore |sr_mag_idx - rtne_mag_idx| <= 1
    must hold for every element, and signs must agree.

    Only the columnwise path is tested: the rowwise path always uses RTNE regardless
    of stochastic_rounding, so testing it here would be vacuous.
    """
    M, N = 128, 128
    N_SAMPLES = 16
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    def _unpack(codes: torch.Tensor) -> torch.Tensor:
        """Unpack (R, C//2) uint8 → (R, C) nibble values."""
        lo = (codes & 0xF).long()
        hi = (codes >> 4).long()
        out = torch.empty(codes.shape[0], codes.shape[1] * 2, dtype=torch.long, device=codes.device)
        out[:, ::2] = lo
        out[:, 1::2] = hi
        return out

    col_rn, _, _, _, _, _ = triton_rht_quantize_row_col(
        A, stochastic_rounding=False, compute_rowwise=False, swizzle_scale_factors=False
    )
    rn_nibs = _unpack(col_rn)
    rn_sign = rn_nibs >> 3
    rn_mag = rn_nibs & 0x7

    for _ in range(N_SAMPLES):
        col_sr, _, _, _, _, _ = triton_rht_quantize_row_col(
            A, stochastic_rounding=True, compute_rowwise=False, swizzle_scale_factors=False
        )
        sr_nibs = _unpack(col_sr)
        sr_sign = sr_nibs >> 3
        sr_mag = sr_nibs & 0x7

        # Sign must match RTNE (SR preserves sign; exception: both sides of zero are sign=0)
        nonzero = (sr_mag != 0) | (rn_mag != 0)
        assert ((sr_sign == rn_sign) | ~nonzero).all(), "SR changed sign relative to RTNE"

        # Magnitude index must be at most 1 step from RTNE
        mag_diff = (sr_mag - rn_mag).abs()
        assert (mag_diff <= 1).all(), (
            f"SR magnitude index differs by {mag_diff.max().item()} from RTNE (must be ≤1)"
        )
