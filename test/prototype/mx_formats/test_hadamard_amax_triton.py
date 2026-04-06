"""Tests for triton_rht_amax (SM90+ kernel)."""
import pytest
import torch

from torchao.utils import is_sm_at_least_90

if is_sm_at_least_90():
    from torchao.prototype.mx_formats.hadamard_amax_triton import triton_rht_amax
    from torchao.prototype.mx_formats.hadamard_utils import get_rht_matrix


# M=32 excluded: all BLOCK_M configs (64, 128) exceed M=32 → all autotune configs fail.
_M_VALUES = [64, 96, 128, 160, 256, 512]
# N=100 excluded: TMA TensorDescriptor requires stride % 16 bytes == 0;
# for bf16 this means N % 8 == 0. N=100 (100*2=200 bytes, 200%16=8) fails.
_N_VALUES = [128, 200, 256, 384, 512, 1024]


@pytest.mark.skipif(not is_sm_at_least_90(), reason="Requires SM90+")
@pytest.mark.parametrize("N", _N_VALUES, ids=lambda n: f"N{n}")
@pytest.mark.parametrize("M", _M_VALUES, ids=lambda m: f"M{m}")
@torch.no_grad()
def test_triton_rht_amax_vs_reference(M, N):
    """triton_rht_amax must match the reference RHT matmul amax exactly (bitwise)."""
    torch.manual_seed(42)
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    # Reference: same deterministic matrix (lru_cached, hard-coded sign vector)
    B = get_rht_matrix(with_random_sign_mask=True, device="cuda")
    ref_amax = (A.t().reshape(N * M // 16, 16) @ B).to(torch.bfloat16).abs().max().float()

    triton_amax = triton_rht_amax(A)

    torch.testing.assert_close(triton_amax, ref_amax, atol=0, rtol=0)
