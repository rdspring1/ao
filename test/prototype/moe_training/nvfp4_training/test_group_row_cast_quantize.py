# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rowwise 1x16 NVFP4 weight quantize. Design doc §11.1.

Test order follows the doc's rule for grouped kernels: prove ``num_tensors = 1``
against the linear reference first, then multi-expert, then group isolation. A kernel
that is wrong only at ``E > 1`` has a group-index bug, not a numerics bug, and the
ordering is what makes that distinction readable from the failure list.
"""

import pytest
import torch
from torch.utils._triton import has_triton

from torchao.utils import is_sm_at_least_100, torch_version_at_least

from ._assertions import assert_codes_bitwise, assert_scales_bitwise
from .nvfp4_reference import reference_row_cast_quantize, reference_weight_quantize_2d

# Flip to True once the @triton.jit body in group_row_cast_quantize_triton.py lands.
# Tests that exercise only the host wrapper -- validation and register_fake -- run
# regardless, because that layer is complete today.
_KERNEL_IMPLEMENTED = False

requires_sm100 = [
    pytest.mark.skipif(not has_triton(), reason="unsupported without triton"),
    pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+"),
    pytest.mark.skipif(
        not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
    ),
]
_requires_kernel = pytest.mark.skipif(
    not _KERNEL_IMPLEMENTED,
    reason="Triton kernel body in group_row_cast_quantize_triton.py is still a stub",
)


def _maybe_sm100(fn):
    for mark in requires_sm100:
        fn = mark(fn)
    return fn


def _needs_kernel(fn):
    return _maybe_sm100(_requires_kernel(fn))


if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
        triton_group_row_cast_quantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
        triton_group_weight_amax,
    )


def _weights(E, M, N, *, seed=0, scale=0.05):
    torch.manual_seed(seed)
    return (torch.randn(E, M, N, device="cuda") * scale).bfloat16()


# ---------------------------------------------------------------------------
# 1. Degeneracy, 2. multi-expert, 3. per-expert isolation
# ---------------------------------------------------------------------------


@_needs_kernel
@torch.no_grad()
def test_single_expert_matches_the_linear_reference():
    """``num_tensors = 1`` is bitwise equal to the plain 1x16 reference.

    This is the case the V1_REQUANT and V2 dense linears actually run, so it is the
    first thing that has to hold.
    """
    W = _weights(1, 256, 512)
    amax = triton_group_weight_amax(W, 1)
    codes, scales = triton_group_row_cast_quantize(W, amax, 1)
    ref = reference_row_cast_quantize(W[0], amax[0])
    assert_codes_bitwise(codes[0], ref.codes, "codes")
    assert_scales_bitwise(scales[0], ref.scales, "scales")


@_needs_kernel
@pytest.mark.parametrize("E", [2, 4, 8])
@torch.no_grad()
def test_multi_expert_matches_the_reference_per_expert(E):
    W = _weights(E, 256, 512)
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    for e in range(E):
        ref = reference_row_cast_quantize(W[e], amax[e])
        assert_codes_bitwise(codes[e], ref.codes, f"codes[{e}]")
        assert_scales_bitwise(scales[e], ref.scales, f"scales[{e}]")


@_needs_kernel
@torch.no_grad()
def test_per_expert_amax_is_not_a_global_reduction():
    """Expert ``g`` must use ``global_amax[g]``.

    One expert is scaled up by 1000x. Under a global reduction every other expert's
    scales would collapse; under per-expert amaxes they are untouched.
    """
    E = 4
    W = _weights(E, 256, 512)
    amax = triton_group_weight_amax(W, E)
    baseline = triton_group_row_cast_quantize(W, amax, E)

    W_hot = W.clone()
    W_hot[1] *= 1000.0
    amax_hot = triton_group_weight_amax(W_hot, E)
    hot = triton_group_row_cast_quantize(W_hot, amax_hot, E)

    for e in range(E):
        if e == 1:
            continue
        assert_codes_bitwise(hot[0][e], baseline[0][e], f"codes[{e}] leaked")
        assert_scales_bitwise(hot[1][e], baseline[1][e], f"scales[{e}] leaked")


@_needs_kernel
@torch.no_grad()
def test_one_zero_expert_leaves_its_neighbours_alone():
    E = 4
    W = _weights(E, 256, 512)
    W[2] = 0.0
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    assert (codes[2] == 0).all(), "zero expert must produce zero codes"
    assert (scales[2].view(torch.uint8) == 0).all(), (
        "zero expert must produce zero scales"
    )
    for e in (0, 1, 3):
        ref = reference_row_cast_quantize(W[e], amax[e])
        assert_codes_bitwise(codes[e], ref.codes, f"codes[{e}]")


# ---------------------------------------------------------------------------
# 4. What makes this op different from the 2D kernel it replaces
# ---------------------------------------------------------------------------


@_needs_kernel
@torch.no_grad()
def test_scale_count_is_1x16_not_16x16():
    """``M * N / 16`` logical scales, not ``M * N / 256``.

    The whole point of replacing the 2D weight quantize: a row no longer shares its
    scale byte with the fifteen rows above and below it.
    """
    E, M, N = 1, 256, 512
    W = _weights(E, M, N)
    amax = triton_group_weight_amax(W, E)
    _, scales = triton_group_row_cast_quantize(W, amax, E)
    assert scales[0].numel() == M * N // 16


@_needs_kernel
@torch.no_grad()
def test_rows_in_one_16x16_tile_get_independent_scales():
    """Under the 2D scheme these two rows were forced to share a scale byte.

    Row 0 is tiny and row 1 is large within the same 16x16 tile; 1x16 scaling must
    give them different scale bytes, which is exactly the resolution the 2D scheme
    was throwing away.
    """
    E, M, N = 1, 256, 512
    W = _weights(E, M, N)
    W[0, 0, :] = 1e-4
    W[0, 1, :] = 1.0
    amax = triton_group_weight_amax(W, E)
    _, scales = triton_group_row_cast_quantize(W, amax, E)
    ref = reference_row_cast_quantize(W[0], amax[0])
    assert_scales_bitwise(scales[0], ref.scales, "scales")
    assert not torch.equal(
        ref.block_scale[0].view(torch.uint8), ref.block_scale[1].view(torch.uint8)
    )


@_needs_kernel
@torch.no_grad()
def test_rowwise_error_is_no_worse_than_the_2d_scheme():
    """Per block, 1x16 error <= 16x16 error -- a finer scale cannot hurt.

    Recorded as a comparison rather than an equality: this is the accuracy claim that
    justifies the extra backward requantization pass, so it should fail loudly if the
    relationship ever inverts.
    """
    from ._assertions import dequantize

    E, M, N = 1, 256, 512
    W = _weights(E, M, N)
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    err_1d = (dequantize(codes[0], scales[0], amax[0]) - W[0].float()).abs()

    ref_2d, _ = reference_weight_quantize_2d(W[0], amax[0])
    err_2d = (dequantize(ref_2d.codes, ref_2d.scales, amax[0]) - W[0].float()).abs()

    block_1d = err_1d.reshape(M, N // 16, 16).sum(-1)
    block_2d = err_2d.reshape(M, N // 16, 16).sum(-1)
    assert (block_1d <= block_2d + 1e-6).all()


# ---------------------------------------------------------------------------
# Wrapper-layer tests -- these run today, no kernel body required
# ---------------------------------------------------------------------------


@_maybe_sm100
@torch.no_grad()
def test_return_arity_is_two_with_no_columnwise_output():
    """Asserted through ``register_fake`` so it holds without a kernel body.

    The columnwise operand is deliberately absent: it is rebuilt in backward by
    §11.6/§11.7 or §11.4/§11.5 from these codes, which is what puts both GEMMs on one
    ``W_qdq``.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 2, 256, 512
    with FakeTensorMode():
        W = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        out = triton_group_row_cast_quantize(W, amax, E)
    assert len(out) == 2
    codes, scales = out
    assert codes.shape == (E, M, N // 2) and codes.dtype == torch.uint8
    assert scales.shape == (E, M // 128, N // 64, 32, 16)
    assert scales.dtype == torch.float8_e4m3fn


@_maybe_sm100
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda W, a, E: (W.float(), a, E), "Expected bfloat16"),
        (lambda W, a, E: (W[0], a, E), "must be 3-D"),
        (lambda W, a, E: (W, a, E + 1), "experts"),
        (lambda W, a, E: (W, a[:1], E), "global_amax must have shape"),
        (lambda W, a, E: (W, a.double(), E), "Expected float32 global_amax"),
        # Contiguous on purpose: a slice would trip the contiguity check first
        # and never reach the divisibility one.
        (lambda W, a, E: (W[:, :100].contiguous(), a, E), "divisible"),
        (lambda W, a, E: (W[:, :100], a, E), "contiguous"),
    ],
)
@torch.no_grad()
def test_validation_rejects_bad_inputs(mutate, message):
    E, M, N = 2, 256, 512
    W = torch.empty(E, M, N, dtype=torch.bfloat16, device="cuda")
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=message):
        triton_group_row_cast_quantize(*mutate(W, amax, E))
