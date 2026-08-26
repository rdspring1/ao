# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rotated columnwise weight requantization. Design doc §11.4 and §11.5.

§11.6/§11.7 with an RHT-128 rotation, for V2's dgrad. The rotation is what lets the
dgrad GEMM cancel: the ``dy`` operand carries ``R_n`` too. One ``dgrad_rht`` is shared
across every expert, which is the grouped-plus-RHT case the implementation must
support.
"""

import pytest
import torch

from ._assertions import assert_codes_bitwise, assert_scales_bitwise
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    reference_col_rht_requant_amax,
    reference_group_col_rht_requant_amax,
    reference_group_col_rht_requantize,
)

# Flip to True once both @triton.jit bodies in group_col_rht_requantize_triton.py land.
_KERNEL_IMPLEMENTED = False
_needs_kernel = kernel_gate(_KERNEL_IMPLEMENTED, "group_col_rht_requantize_triton.py")

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.group_col_rht_requantize_triton import (
        triton_group_col_rht_requant_amax,
        triton_group_col_rht_requantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_quantize_triton import (
        triton_group_row_cast_quantize,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_weight_amax_triton import (
        triton_group_weight_amax,
    )


def _signs(device="cuda", seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _packed_weights(E, M, N, *, seed=0, scale=0.05):
    torch.manual_seed(seed)
    W = (torch.randn(E, M, N, device="cuda") * scale).bfloat16()
    amax = triton_group_weight_amax(W, E)
    codes, scales = triton_group_row_cast_quantize(W, amax, E)
    return W, codes, scales, amax


# --- §11.4 ------------------------------------------------------------------


@_needs_kernel
@torch.no_grad()
def test_amax_single_expert_matches_reference():
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    got = triton_group_col_rht_requant_amax(codes, scales, amax, d, 1)
    want = reference_col_rht_requant_amax(codes[0], scales[0], amax[0], d)
    torch.testing.assert_close(got[0], want, rtol=1e-3, atol=1e-3)


@_needs_kernel
@pytest.mark.parametrize("E", [2, 4])
@torch.no_grad()
def test_amax_multi_expert_matches_reference(E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    got = triton_group_col_rht_requant_amax(codes, scales, amax, d, E)
    want = reference_group_col_rht_requant_amax(codes, scales, amax, d)
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)


@_needs_kernel
@torch.no_grad()
def test_amax_changes_with_the_sign_vector():
    """A different rotation gives a different bound; the signs really reach the kernel."""
    _, codes, scales, amax = _packed_weights(1, 256, 512)
    a = triton_group_col_rht_requant_amax(codes, scales, amax, _signs(seed=0), 1)
    b = triton_group_col_rht_requant_amax(codes, scales, amax, _signs(seed=1), 1)
    assert a[0].item() != b[0].item()


@_needs_kernel
@torch.no_grad()
def test_amax_is_computed_from_the_quantized_weight():
    """Discriminating test that the lazy path consumes ``row_fp4_w``, not the BF16 W.

    Computing the same quantity from the original weight gives a different answer
    once the rotation is involved, because ``W`` and ``W_qdq`` differ per element even
    though their maxima coincide (see the note in the unrotated twin's test file).
    """
    from .nvfp4_reference import reference_dynamic_rht

    W, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    got = triton_group_col_rht_requant_amax(codes, scales, amax, d, 1)
    from_bf16 = reference_dynamic_rht(W[0], d, transpose=True).float().abs().max()
    assert abs(got[0].item() - from_bf16.item()) > 1e-6, (
        "an amax taken over the bf16 weight would not match; the op must measure W_qdq"
    )


@_needs_kernel
@torch.no_grad()
def test_amax_per_expert_isolation_and_shared_sign_vector():
    """One ``B`` operand serves every expert, and no expert's amax leaks into another."""
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    baseline = triton_group_col_rht_requant_amax(codes, scales, amax, d, E)
    hot = amax.clone()
    hot[2] *= 1000.0
    got = triton_group_col_rht_requant_amax(codes, scales, hot, d, E)
    for e in range(E):
        if e == 2:
            continue
        assert got[e].item() == baseline[e].item(), f"expert {e} leaked"


# --- §11.5 ------------------------------------------------------------------


@_needs_kernel
@pytest.mark.parametrize("E", [1, 2, 4])
@torch.no_grad()
def test_requantize_matches_reference(E):
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    amax_t = triton_group_col_rht_requant_amax(codes, scales, amax, d, E)
    got_codes, got_scales = triton_group_col_rht_requantize(
        codes, scales, amax, amax_t, d, E
    )
    refs = reference_group_col_rht_requantize(codes, scales, amax, amax_t, d)
    for e in range(E):
        assert_codes_bitwise(got_codes[e], refs[e].codes, f"codes[{e}]")
        assert_scales_bitwise(got_scales[e], refs[e].scales, f"scales[{e}]")


@_needs_kernel
@torch.no_grad()
def test_a_halved_amax_saturates_only_that_expert():
    E = 4
    _, codes, scales, amax = _packed_weights(E, 256, 512)
    d = _signs()
    amax_t = triton_group_col_rht_requant_amax(codes, scales, amax, d, E)
    baseline = triton_group_col_rht_requantize(codes, scales, amax, amax_t, d, E)
    halved = amax_t.clone()
    halved[1] *= 0.5
    got = triton_group_col_rht_requantize(codes, scales, amax, halved, d, E)
    assert not torch.equal(got[0][1], baseline[0][1])
    for e in (0, 2, 3):
        assert_codes_bitwise(got[0][e], baseline[0][e], f"codes[{e}] leaked")


@_needs_kernel
@torch.no_grad()
def test_decode_numerator_is_2688_not_1536():
    """The weight is a *cast* operand even in V2; only MS-EDEN operands use 1536.

    Checked through the reference, which is parameterized on the FP8 ceiling: decoding
    with the MS-EDEN numerator must disagree, so the test is sensitive to the mistake
    the design doc calls out as "backward off by roughly 40%".
    """
    from .nvfp4_reference import EDEN_BLOCK_SCALE_MAX, reference_dequantize_rowwise

    _, codes, scales, amax = _packed_weights(1, 256, 512)
    d = _signs()
    amax_t = triton_group_col_rht_requant_amax(codes, scales, amax, d, 1)
    col_codes, col_scales = triton_group_col_rht_requantize(
        codes, scales, amax, amax_t, d, 1
    )
    right = reference_dequantize_rowwise(col_codes[0], col_scales[0], amax_t[0])
    wrong = reference_dequantize_rowwise(
        col_codes[0], col_scales[0], amax_t[0], fp8_max=EDEN_BLOCK_SCALE_MAX
    )
    assert not torch.allclose(right, wrong, rtol=1e-3, atol=1e-6)


# --- wrapper layer, runs today ----------------------------------------------


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
        d = torch.empty(128, dtype=torch.int8, device="cuda")
        got = triton_group_col_rht_requant_amax(codes, scales, amax, d, E)
        assert got.shape == (E,) and got.dtype == torch.float32
        col_codes, col_scales = triton_group_col_rht_requantize(
            codes, scales, amax, got, d, E
        )
    assert col_codes.shape == (E, N, M // 2)
    assert col_scales.shape == (E, N // 128, M // 64, 32, 16)


@maybe_sm100
@pytest.mark.parametrize("bad_len", [16, 64, 256])
@torch.no_grad()
def test_rejects_a_non_128_sign_vector(bad_len):
    """V2 is RHT-128 only. A 16-element vector is V1's and must not be accepted here."""
    E, M, N = 2, 256, 512
    codes = torch.zeros(E, M, N // 2, dtype=torch.uint8, device="cuda")
    scales = torch.zeros(
        E, M // 128, N // 64, 32, 16, dtype=torch.float8_e4m3fn, device="cuda"
    )
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    bad = torch.ones(bad_len, dtype=torch.int8, device="cuda")
    with pytest.raises(ValueError, match=r"dgrad_rht must be a \(128,\) tensor"):
        triton_group_col_rht_requant_amax(codes, scales, amax, bad, E)
