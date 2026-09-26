# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""RHT matrix construction for both Hadamard sizes. Design doc §18 step 1.

Runs today: it needs no Triton kernel, only ``hadamard_utils``. Everything the
recipes rely on about the transform is pinned here rather than inside a kernel test,
because a wrong Hadamard makes every downstream kernel wrong in the same way and the
failure would otherwise surface as a diffuse accuracy loss.
"""

import math

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    DEFAULT_SIGN_VECTOR,
    get_dynamic_rht_matrix,
    get_hadamard_matrix,
    get_rht_matrix,
)

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def _signs(n: int, device="cpu", seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


@pytest.mark.parametrize("dim", [16, 128])
def test_hadamard_is_orthonormal(dim):
    """``H @ H.T == I``. Float64 so the assertion is about the construction, not bf16."""
    H = get_hadamard_matrix(dim, "cpu", torch.float64)
    assert H.shape == (dim, dim)
    torch.testing.assert_close(
        H @ H.T, torch.eye(dim, dtype=torch.float64), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("dim", [16, 128])
def test_hadamard_is_normalized(dim):
    """Every entry is exactly ``+-1/sqrt(dim)``.

    The normalization is load-bearing on both operands of a GEMM: without it the
    product comes back scaled by ``dim``, which is the failure the design doc calls
    out as "non-normalized Hadamard".
    """
    H = get_hadamard_matrix(dim, "cpu", torch.float64)
    torch.testing.assert_close(
        H.abs(),
        torch.full_like(H, 1.0 / math.sqrt(dim)),
        atol=1e-12,
        rtol=0,
    )


def test_hadamard_128_contains_16_as_a_sylvester_block():
    """H128 is ``kron(H16, H16[:8,:8])``, matching the V2 reference's popcount order.

    Pinned explicitly because any other Hadamard of order 128 is also orthonormal and
    normalized -- the two properties above cannot tell them apart, but a kernel built
    on one and an oracle built on another disagree everywhere.
    """
    h16 = get_hadamard_matrix(16, "cpu", torch.float64) * math.sqrt(16)
    h128 = get_hadamard_matrix(128, "cpu", torch.float64) * math.sqrt(128)
    torch.testing.assert_close(torch.kron(h16, h16[:8, :8]), h128, atol=1e-12, rtol=0)


def test_unsupported_hadamard_dimension_raises():
    with pytest.raises(ValueError, match="16 or 128"):
        get_hadamard_matrix(64, "cpu", torch.float32)


@pytest.mark.parametrize("dim", [16, 128])
def test_dynamic_rht_is_orthonormal(dim):
    """``R = diag(signs) @ H`` stays orthonormal -- the signs cannot break cancellation."""
    R = get_dynamic_rht_matrix(_signs(dim), torch.float64)
    torch.testing.assert_close(
        R @ R.T, torch.eye(dim, dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_dynamic_matches_cached_for_the_same_signs():
    """The dynamic and cached builders must agree bitwise at RHT-16.

    V1_REQUANT resolves its matrix through the cached path and V2 through the dynamic
    one; if they disagreed, a test written against one recipe would not transfer.
    """
    signs = torch.tensor(DEFAULT_SIGN_VECTOR, dtype=torch.int8)
    dynamic = get_dynamic_rht_matrix(signs, torch.float64)
    cached = get_rht_matrix(DEFAULT_SIGN_VECTOR, "cpu", torch.float64, 16)
    torch.testing.assert_close(dynamic, cached, atol=0, rtol=0)


def test_all_positive_signs_reduce_to_the_plain_hadamard():
    ones = torch.ones(128, dtype=torch.int8)
    torch.testing.assert_close(
        get_dynamic_rht_matrix(ones, torch.float64),
        get_hadamard_matrix(128, "cpu", torch.float64),
        atol=0,
        rtol=0,
    )


def test_different_signs_give_different_matrices():
    a = get_dynamic_rht_matrix(_signs(128, seed=0), torch.float64)
    b = get_dynamic_rht_matrix(_signs(128, seed=1), torch.float64)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("bad", [torch.zeros(64), torch.zeros(2, 128)])
def test_dynamic_rht_rejects_bad_sign_shapes(bad):
    with pytest.raises(ValueError):
        get_dynamic_rht_matrix(bad.to(torch.int8), torch.float32)


def test_dynamic_rht_does_not_leak_one_entry_per_resample():
    """The regression test for V2's documented memory-growth failure mode.

    Caching ``diag(signs) @ H`` by sign value grows the cache once per resample, for
    the lifetime of the run. ``get_dynamic_rht_matrix`` must memoize only the fixed
    Hadamard, so the cache size is flat no matter how many vectors pass through.
    """
    get_dynamic_rht_matrix(_signs(128), torch.float32)
    before = get_hadamard_matrix.cache_info().currsize
    for seed in range(256):
        get_dynamic_rht_matrix(_signs(128, seed=seed), torch.float32)
    assert get_hadamard_matrix.cache_info().currsize == before


def test_cached_builder_rejects_a_tensor():
    """A live V2 sign buffer handed to V1's cached helper must fail loudly.

    An unhashable argument is the reason this raises, and that is the desired
    behaviour: the alternative -- silently succeeding -- would be the memory leak
    above. The design doc lists ``TypeError: unhashable type: 'Tensor'`` as the
    signature of this mistake.
    """
    with pytest.raises(TypeError):
        get_rht_matrix(torch.ones(16, dtype=torch.int8), "cpu", torch.float32, 16)


@_requires_cuda
def test_rht_cancels_without_quantization():
    """§13/§14 in exact arithmetic: ``(dy @ R) @ (w.t() @ R).t() == dy @ w``.

    No quantizer involved, so any failure here is the transform itself -- the axis,
    the transpose, the sign vector, or the ``1/sqrt(n)`` factor.
    """
    torch.manual_seed(0)
    dy = torch.randn(256, 512, device="cuda", dtype=torch.float64)
    w = torch.randn(384, 512, device="cuda", dtype=torch.float64)
    R = get_dynamic_rht_matrix(_signs(128, device="cuda"), torch.float64)

    lhs = dy.reshape(-1, 128) @ R
    rhs = w.reshape(-1, 128) @ R
    got = lhs.reshape(dy.shape) @ rhs.reshape(w.shape).t()
    torch.testing.assert_close(got, dy @ w.t(), rtol=1e-10, atol=1e-10)


@_requires_cuda
def test_a_crossed_sign_vector_breaks_cancellation():
    """The discriminating test for wiring ``dgrad_rht`` where ``wgrad_rht`` belongs.

    Rotating the two operands with different sign vectors leaves a residual instead
    of the identity, and produces no error of its own -- only a wrong gradient.
    """
    torch.manual_seed(0)
    dy = torch.randn(256, 512, device="cuda", dtype=torch.float64)
    w = torch.randn(384, 512, device="cuda", dtype=torch.float64)
    R_a = get_dynamic_rht_matrix(_signs(128, device="cuda", seed=0), torch.float64)
    R_b = get_dynamic_rht_matrix(_signs(128, device="cuda", seed=1), torch.float64)

    lhs = (dy.reshape(-1, 128) @ R_a).reshape(dy.shape)
    rhs = (w.reshape(-1, 128) @ R_b).reshape(w.shape)
    assert not torch.allclose(lhs @ rhs.t(), dy @ w.t(), rtol=1e-3, atol=1e-3)


@_requires_cuda
def test_unnormalized_hadamard_scales_the_product_by_n():
    """Why ``1/sqrt(n)`` is required on both sides, not just consistently."""
    torch.manual_seed(0)
    dy = torch.randn(256, 512, device="cuda", dtype=torch.float64)
    w = torch.randn(384, 512, device="cuda", dtype=torch.float64)
    R = get_dynamic_rht_matrix(_signs(128, device="cuda"), torch.float64) * math.sqrt(
        128
    )
    lhs = (dy.reshape(-1, 128) @ R).reshape(dy.shape)
    rhs = (w.reshape(-1, 128) @ R).reshape(w.shape)
    torch.testing.assert_close(lhs @ rhs.t(), 128 * (dy @ w.t()), rtol=1e-10, atol=1e-8)
