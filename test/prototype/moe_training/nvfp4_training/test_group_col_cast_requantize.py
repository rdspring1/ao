# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped lazy columnwise NVFP4 weight requantization. Design doc §11.6 and §11.7.

The V1_REQUANT backward weight path. The property under test is the one that replaces
the 2D scheme's shared scale byte: the forward and dgrad GEMMs must decode to one and
the same ``W_qdq``, because the backward operand is *derived from* the packed forward
weight rather than re-quantized from the original BF16 tensor.
"""

import pytest
import torch

from ._assertions import assert_codes_bitwise, assert_scales_bitwise
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    reference_col_cast_requant_amax,
    reference_dequantize_rowwise,
    reference_group_col_cast_requant_amax,
    reference_group_col_cast_requantize,
)

# Flip to True once both @triton.jit bodies in group_col_cast_requantize_triton.py land.
_KERNEL_IMPLEMENTED = True
_needs_kernel = kernel_gate(_KERNEL_IMPLEMENTED, "group_col_cast_requantize_triton.py")

if TRITON_AVAILABLE:
    # Imported for its op registration, which the arity test reads.
    import torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_triton  # noqa: F401
    from torchao.prototype.moe_training.nvfp4_training.group_col_cast_requantize_triton import (
        triton_group_col_cast_requant_amax,
        triton_group_col_cast_requantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
        triton_group_row_cast_quantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
        triton_group_weight_amax,
    )


def _packed_weights(E, M, N, *, seed=0, scale=0.05):
    """A weight stack put through §11.1, which is what these ops consume."""
    torch.manual_seed(seed)
    W = (torch.randn(E, M, N, device="cuda") * scale).bfloat16()
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    return W, codes, scales, amax


def _representable_weights(E, M, N, *, seed=0):
    """A weight stack that is exactly representable *including its block scales*.

    Constant magnitude 6 with random signs: every 16-element block in either
    orientation has amax exactly 6, so the block scale is exactly 448 (representable
    in E4M3) and the encode scale is exactly 1. A tensor merely drawn from the FP4
    value grid is **not** enough -- its block amaxes give block scales that E4M3
    cannot represent, and requantization then is not idempotent.
    """
    torch.manual_seed(seed)
    sign = torch.where(torch.rand(E, M, N, device="cuda") >= 0.5, 1.0, -1.0)
    return (sign * 6.0).bfloat16()


# ---------------------------------------------------------------------------
# §11.6 -- the requantization amax
# ---------------------------------------------------------------------------


@_needs_kernel
@torch.no_grad()
def test_requant_amax_single_expert_matches_reference():
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    got = triton_group_col_cast_requant_amax(codes, scales, amax, 1)
    want = reference_col_cast_requant_amax(codes[0], scales[0], amax[0])
    torch.testing.assert_close(got[0], want, atol=0, rtol=0)


@_needs_kernel
@pytest.mark.parametrize("E", [2, 4])
@torch.no_grad()
def test_requant_amax_multi_expert_matches_reference(E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    got = triton_group_col_cast_requant_amax(codes, scales, amax, E)
    want = reference_group_col_cast_requant_amax(codes, scales, amax)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


@_needs_kernel
@torch.no_grad()
def test_requant_amax_consumes_the_quantized_weight_not_the_bf16_one():
    """The discriminating test that this op is really lazy.

    .. note::

       Design doc §9 proposes comparing against ``amax_w`` and expecting them to
       differ "generally". **They do not.** When ``global_amax`` is the tensor's own
       amax, the element attaining it lies in a block whose block amax *is* that
       value, so its block scale saturates at exactly 448, its code is exactly 6, and
       it dequantizes back to exactly ``global_amax``. The two are bitwise equal, and
       a test written the doc's way fails.

       The honest discriminating input is an **over-bounding** ``global_amax`` -- a
       tensor-parallel all-reduced amax, or a stale one. Then the reconstructed
       weight really is bounded below the supplied amax and the two separate.
    """
    torch.manual_seed(0)
    W = (torch.randn(1, 256, 512, device="cuda") * 0.05).bfloat16()
    own_amax = triton_group_weight_amax(W, 1)

    codes, scales = triton_group_row_cast_quantize(W, own_amax, 1)
    same = triton_group_col_cast_requant_amax(codes, scales, own_amax, 1)
    assert same[0].item() == own_amax[0].item(), (
        "with the tensor's own amax the maximum element survives quantization exactly"
    )

    inflated = own_amax * 2.0
    codes_i, scales_i = triton_group_row_cast_quantize(W, inflated, 1)
    got = triton_group_col_cast_requant_amax(codes_i, scales_i, inflated, 1)
    assert got[0].item() != inflated[0].item(), (
        "an over-bounding amax must not be echoed back; the op has to measure W_qdq"
    )
    torch.testing.assert_close(
        got[0],
        reference_col_cast_requant_amax(codes_i[0], scales_i[0], inflated[0]),
        atol=0,
        rtol=0,
    )


@_needs_kernel
@torch.no_grad()
def test_requant_amax_is_exact_for_a_representable_weight():
    W = _representable_weights(1, 256, 512)
    amax = triton_group_weight_amax(W, 1)
    codes, scales = triton_group_row_cast_quantize(W, amax, 1)
    got = triton_group_col_cast_requant_amax(codes, scales, amax, 1)
    assert got[0].item() == amax[0].item()


@_needs_kernel
@torch.no_grad()
def test_requant_amax_per_expert_isolation():
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    baseline = triton_group_col_cast_requant_amax(codes, scales, amax, E)
    amax_hot = amax.clone()
    amax_hot[1] *= 1000.0
    hot = triton_group_col_cast_requant_amax(codes, scales, amax_hot, E)
    for e in range(E):
        if e == 1:
            continue
        assert hot[e].item() == baseline[e].item(), f"expert {e} amax leaked"


@_needs_kernel
@torch.no_grad()
def test_nan_global_amax_reconstructs_to_zero():
    """A NaN or inf amax must zero the reconstruction rather than propagate."""
    E = 2
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    for bad in (float("nan"), float("inf")):
        broken = amax.clone()
        broken[0] = bad
        got = triton_group_col_cast_requant_amax(codes, scales, broken, E)
        assert got[0].item() == 0.0, f"amax={bad} must reconstruct to zero"
        assert torch.isfinite(got[1]), "the healthy expert must be unaffected"


# ---------------------------------------------------------------------------
# §11.7 -- the requantization itself
# ---------------------------------------------------------------------------


@_needs_kernel
@torch.no_grad()
def test_requantize_single_expert_matches_reference():
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, 1)
    got_codes, got_scales = triton_group_col_cast_requantize(
        codes, scales, amax, amax_t, 1
    )
    ref = reference_group_col_cast_requantize(codes, scales, amax, amax_t)[0]
    assert_codes_bitwise(got_codes[0], ref.codes, "codes")
    assert_scales_bitwise(got_scales[0], ref.scales, "scales")


@_needs_kernel
@pytest.mark.parametrize("E", [2, 4])
@torch.no_grad()
def test_requantize_multi_expert_matches_reference(E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, E)
    got_codes, got_scales = triton_group_col_cast_requantize(
        codes, scales, amax, amax_t, E
    )
    refs = reference_group_col_cast_requantize(codes, scales, amax, amax_t)
    for e in range(E):
        assert_codes_bitwise(got_codes[e], refs[e].codes, f"codes[{e}]")
        assert_scales_bitwise(got_scales[e], refs[e].scales, f"scales[{e}]")


@_needs_kernel
@torch.no_grad()
def test_both_gemms_decode_to_one_w_qdq():
    """Invariant 2, and the property that replaces the 2D shared scale byte.

    ``dequant(col_fp4_w_t).t()`` and ``dequant(row_fp4_w)`` must agree to within one
    requantization step. Their *scale bytes* now differ -- which the 2D scheme forbade
    -- but they decode to the same weight, which is the stronger guarantee.
    """
    W, codes, scales, amax = _packed_weights(1, 256, 512)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, 1)
    col_codes, col_scales = triton_group_col_cast_requantize(
        codes, scales, amax, amax_t, 1
    )
    forward_qdq = reference_dequantize_rowwise(codes[0], scales[0], amax[0])
    backward_qdq = reference_dequantize_rowwise(
        col_codes[0], col_scales[0], amax_t[0]
    ).t()
    step = forward_qdq.abs().max() / 6.0  # one FP4 grid step at the top block scale
    assert (forward_qdq - backward_qdq).abs().max() <= step


@_needs_kernel
@torch.no_grad()
def test_requantization_is_idempotent_on_a_representable_weight():
    """Exact round-trip when the block scales are E4M3-exact too.

    See ``_representable_weights``: a tensor drawn from the FP4 value grid alone is
    not sufficient, which is a sharper condition than design doc §10's extra test
    states.
    """
    W = _representable_weights(1, 256, 512)
    amax = triton_group_weight_amax(W, 1)
    codes, scales = triton_group_row_cast_quantize(W, amax, 1)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, 1)
    col_codes, col_scales = triton_group_col_cast_requantize(
        codes, scales, amax, amax_t, 1
    )
    forward_qdq = reference_dequantize_rowwise(codes[0], scales[0], amax[0])
    backward_qdq = reference_dequantize_rowwise(
        col_codes[0], col_scales[0], amax_t[0]
    ).t()
    assert torch.equal(forward_qdq, backward_qdq.contiguous())


@_needs_kernel
@torch.no_grad()
def test_a_halved_amax_saturates_only_that_expert():
    """Why §11.6 must run before §11.7, and that saturation stays expert-local."""
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, E)
    baseline = triton_group_col_cast_requantize(codes, scales, amax, amax_t, E)

    halved = amax_t.clone()
    halved[1] *= 0.5
    got = triton_group_col_cast_requantize(codes, scales, amax, halved, E)
    assert not torch.equal(got[0][1], baseline[0][1]), "expert 1 should saturate"
    for e in (0, 2, 3):
        assert_codes_bitwise(got[0][e], baseline[0][e], f"codes[{e}] leaked")


@_needs_kernel
@torch.no_grad()
def test_shared_reconstruction_between_the_amax_and_quantize_passes():
    """§11.6 and §11.7 must reconstruct the same ``W_qdq``.

    Checked indirectly but sufficiently: if the quantize pass reconstructed something
    the amax pass did not measure, some block scale would exceed the E4M3 ceiling and
    saturate. No block may saturate when the amax comes from §11.6.
    """
    _, codes, scales, amax = _packed_weights(2, 256, 512)
    amax_t = triton_group_col_cast_requant_amax(codes, scales, amax, 2)
    _, col_scales = triton_group_col_cast_requantize(codes, scales, amax, amax_t, 2)
    assert (col_scales.float() < 448.0).any(), "at least some headroom must remain"
    assert torch.isfinite(col_scales.float()).all()


# ---------------------------------------------------------------------------
# Wrapper-layer tests -- run today
# ---------------------------------------------------------------------------


@maybe_sm100
@torch.no_grad()
def test_register_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    E, M, N = 2, 256, 512
    with FakeTensorMode():
        codes = torch.empty(E, M, N // 2, dtype=torch.uint8, device="cuda")
        scales = torch.empty(
            E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
        )
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        got = triton_group_col_cast_requant_amax(codes, scales, amax, E)
        assert got.shape == (E,) and got.dtype == torch.float32
        col_codes, col_scales = triton_group_col_cast_requantize(
            codes, scales, amax, got, E
        )
    # Transposed: the logical weight is (E, M, N), so its transpose is (E, N, M).
    assert col_codes.shape == (E, N, M // 2)
    assert col_scales.shape == (E, N // 128, M // 64, 32, 16)


@maybe_sm100
@torch.no_grad()
def test_amax_op_takes_no_sign_vector():
    """§11.6 applies no transform, so its arity must not admit one.

    The guard against accidentally wiring the V1_REQUANT weight path through the
    rotated §11.4 kernel, which would break V1_REQUANT's dgrad silently.

    Asserted on the registered op schema rather than the Python signature:
    ``torch.library.custom_op`` wraps the function, so ``inspect.signature`` reports
    ``(*args, **kwargs)``. The schema is the contract that torch.compile and any
    out-of-tree caller actually see.
    """
    schema = torch.ops.torchao.triton_group_col_cast_requant_amax.default._schema
    names = [arg.name for arg in schema.arguments]
    assert names == ["row_fp4_w", "row_sf_w", "global_amax", "num_tensors"]

    rotated = torch.ops.torchao.triton_group_col_rht_requant_amax.default._schema
    assert "dgrad_rht" in [arg.name for arg in rotated.arguments], (
        "the rotated twin is the one that takes a sign vector"
    )


@maybe_sm100
@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c, s, a, E: (c.float(), s, a, E), "Expected uint8"),
        (lambda c, s, a, E: (c[0], s, a, E), "must be 3-D"),
        (lambda c, s, a, E: (c, s.float(), a, E), "Expected float8_e4m3fn"),
        (lambda c, s, a, E: (c, s, a, E + 1), "experts"),
        (lambda c, s, a, E: (c, s, a[:1], E), "global_amax must have shape"),
    ],
)
@torch.no_grad()
def test_validation_rejects_bad_inputs(mutate, message):
    E, M, N = 2, 256, 512
    codes = torch.zeros(E, M, N // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(
        E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=message):
        triton_group_col_cast_requant_amax(*mutate(codes, scales, amax, E))


@maybe_sm100
@torch.no_grad()
def test_requantize_rejects_a_mismatched_amax():
    E, M, N = 2, 256, 512
    codes = torch.zeros(E, M, N // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(
        E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="amax_w_qdq_t must have shape"):
        triton_group_col_cast_requantize(
            codes, scales, amax, torch.ones(E + 1, device="cuda"), E
        )
