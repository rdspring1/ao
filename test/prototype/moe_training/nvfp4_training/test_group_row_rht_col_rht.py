# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped RHT-128 on both axes: amax and MS-EDEN quantize. Design doc §11.2, §11.3.

V2's backward gradient path. What distinguishes it from §11.8/§11.9 is that **both**
axes are transformed with **independent** sign vectors. A crossed pair produces no
error, only a wrong gradient, so the tests that separate the two vectors are the
important ones here.
"""

import pytest
import torch

from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    reference_group_row_rht_col_rht_amax,
    reference_row_rht_col_rht_amax,
)

_AMAX_IMPLEMENTED = False
# MS-EDEN also needs `stochastic_rounding_fp8_e4m3` and `_quantize_ms_eden` ported
# from the monorepo; see the kernel module docstring.
_MS_EDEN_IMPLEMENTED = False
_needs_amax = kernel_gate(_AMAX_IMPLEMENTED, "group_row_rht_col_rht_amax_triton.py")
_needs_ms_eden = kernel_gate(
    _AMAX_IMPLEMENTED and _MS_EDEN_IMPLEMENTED,
    "group_row_rht_col_rht_quantize_ms_eden_triton.py",
)

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        VARYING_FIRST_DIM,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_amax_triton import (
        triton_group_row_rht_col_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_rht_col_rht_quantize_ms_eden_triton import (
        triton_group_row_rht_col_rht_quantize_ms_eden,
    )


def _signs(device="cuda", seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _packed(group_sizes, hidden, *, seed=0):
    torch.manual_seed(seed)
    dy = torch.randn(sum(group_sizes), hidden, device="cuda", dtype=torch.bfloat16)
    offs = torch.cumsum(
        torch.tensor(group_sizes, dtype=torch.int32, device="cuda"),
        0,
        dtype=torch.int32,
    )
    return dy, offs


def _amax(dy, d, w, offs, E):
    return triton_group_row_rht_col_rht_amax(
        dy, d, w, offs, E, dy.shape[0], dy.shape[1], VARYING_FIRST_DIM, offs[-1:]
    )


# --- §11.2 ------------------------------------------------------------------


@_needs_amax
@torch.no_grad()
def test_amax_single_group_matches_the_linear_reference():
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    got_row, got_col = _amax(dy, d, w, offs, 1)
    ref_row, ref_col = reference_row_rht_col_rht_amax(dy, d, w)
    torch.testing.assert_close(got_row[0], ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col[0], ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@pytest.mark.parametrize("group_sizes", [[128, 128], [256, 128, 384, 128]])
@torch.no_grad()
def test_amax_multi_group_matches_the_reference(group_sizes):
    dy, offs = _packed(group_sizes, 512)
    d, w = _signs(seed=0), _signs(seed=1)
    E = len(group_sizes)
    got_row, got_col = _amax(dy, d, w, offs, E)
    ref_row, ref_col = reference_group_row_rht_col_rht_amax(dy, d, w, offs, E)
    torch.testing.assert_close(got_row, ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col, ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@torch.no_grad()
def test_each_sign_vector_drives_exactly_one_output():
    """Changing ``dgrad_rht`` may move only the rowwise amax, and vice versa.

    The cleanest statement that the two are not crossed: if the kernel wired them the
    other way round, each assertion below would fail in the opposite direction.
    """
    dy, offs = _packed([256], 512)
    d0, w0 = _signs(seed=0), _signs(seed=1)
    base_row, base_col = _amax(dy, d0, w0, offs, 1)

    row_only, col_unchanged = _amax(dy, _signs(seed=2), w0, offs, 1)
    assert row_only[0].item() != base_row[0].item(), "dgrad_rht must move amax_rht_dy"
    assert col_unchanged[0].item() == base_col[0].item(), (
        "dgrad_rht must not touch amax_rht_dy_t"
    )

    row_unchanged, col_only = _amax(dy, d0, _signs(seed=3), offs, 1)
    assert col_only[0].item() != base_col[0].item(), "wgrad_rht must move amax_rht_dy_t"
    assert row_unchanged[0].item() == base_row[0].item(), (
        "wgrad_rht must not touch amax_rht_dy"
    )


@_needs_amax
@torch.no_grad()
def test_swapping_the_two_sign_vectors_changes_both_outputs():
    """The discriminating test for a crossed-argument bug."""
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    a_row, a_col = _amax(dy, d, w, offs, 1)
    b_row, b_col = _amax(dy, w, d, offs, 1)
    assert a_row[0].item() != b_row[0].item()
    assert a_col[0].item() != b_col[0].item()


@_needs_amax
@torch.no_grad()
def test_identical_sign_vectors_are_not_assumed_equal():
    """``dgrad_rht == wgrad_rht`` must still compute each axis on its own data."""
    dy, offs = _packed([256], 384)
    d = _signs(seed=0)
    got_row, got_col = _amax(dy, d, d, offs, 1)
    ref_row, ref_col = reference_row_rht_col_rht_amax(dy, d, d)
    torch.testing.assert_close(got_row[0], ref_row, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(got_col[0], ref_col, rtol=1e-3, atol=1e-3)


@_needs_amax
@torch.no_grad()
def test_amax_per_group_isolation():
    group_sizes = [128, 128, 128, 128]
    dy, offs = _packed(group_sizes, 512)
    d, w = _signs(seed=0), _signs(seed=1)
    base = _amax(dy, d, w, offs, 4)
    dy2 = dy.clone()
    dy2[256:384] *= 1000.0
    hot = _amax(dy2, d, w, offs, 4)
    for g in (0, 1, 3):
        assert hot[0][g].item() == base[0][g].item(), f"group {g} row leaked"
        assert hot[1][g].item() == base[1][g].item(), f"group {g} col leaked"


@_needs_amax
@torch.no_grad()
def test_zero_gradient_gives_zero_amaxes_without_nan():
    dy, offs = _packed([128, 128], 512)
    got = _amax(torch.zeros_like(dy), _signs(0), _signs(1), offs, 2)
    assert torch.equal(torch.stack(got), torch.zeros(2, 2, device="cuda"))


# --- §11.3 ------------------------------------------------------------------


@_needs_ms_eden
@torch.no_grad()
def test_return_order_is_columnwise_first():
    """§11.3 returns the wgrad pair first, unlike every sibling quantize op.

    Pinned by shape on a non-square input, where a swapped unpack is a shape error.
    On a square layer it would corrupt silently, which is why this is asserted rather
    than left to the type checker.
    """
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(dy, d, w, offs, 1)
    col_codes, col_sf, row_codes, row_sf = (
        triton_group_row_rht_col_rht_quantize_ms_eden(
            dy,
            ar,
            ac,
            d,
            w,
            offs,
            1,
            256,
            512,
            VARYING_FIRST_DIM,
            torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda"),
            offs[-1:],
        )
    )
    assert col_codes.shape == (512, 128), "columnwise operand must come first"
    assert row_codes.shape == (256, 256), "rowwise operand must come second"


@_needs_ms_eden
@torch.no_grad()
def test_block_scales_never_exceed_the_eden_ceiling():
    """MS-EDEN caps block scales at 256, not 448 -- the reason its numerator is 1536."""
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(dy, d, w, offs, 1)
    _, col_sf, _, row_sf = triton_group_row_rht_col_rht_quantize_ms_eden(
        dy,
        ar,
        ac,
        d,
        w,
        offs,
        1,
        256,
        512,
        VARYING_FIRST_DIM,
        torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda"),
        offs[-1:],
    )
    assert col_sf.float().max() <= 256.0
    assert row_sf.float().max() <= 256.0


@_needs_ms_eden
@torch.no_grad()
def test_fixed_rng_state_reproduces_bitwise():
    dy, offs = _packed([256], 512)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(dy, d, w, offs, 1)
    rng = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device="cuda")
    args = (dy, ar, ac, d, w, offs, 1, 256, 512, VARYING_FIRST_DIM, rng, offs[-1:])
    a = triton_group_row_rht_col_rht_quantize_ms_eden(*args)
    b = triton_group_row_rht_col_rht_quantize_ms_eden(*args)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@_needs_ms_eden
@torch.no_grad()
def test_ms_eden_is_unbiased():
    """The property MS-EDEN exists for: ``E[dequant(q(v))] == v`` over the RNG.

    Averaged over many draws, the dequantized rowwise operand must converge on
    ``dy @ R_n``. A biased quantizer would show a systematic offset that no number of
    draws removes.
    """
    from .nvfp4_reference import (
        EDEN_BLOCK_SCALE_MAX,
        reference_dequantize_rowwise,
        reference_dynamic_rht,
    )

    dy, offs = _packed([128], 256)
    d, w = _signs(seed=0), _signs(seed=1)
    ar, ac = _amax(dy, d, w, offs, 1)
    target = reference_dynamic_rht(dy, d, transpose=False).float()

    draws = 64
    total = torch.zeros_like(target)
    for i in range(draws):
        rng = torch.tensor([1, i, 2, i + 1000], dtype=torch.int64, device="cuda")
        _, _, row_codes, row_sf = triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, ar, ac, d, w, offs, 1, 128, 256, VARYING_FIRST_DIM, rng, offs[-1:]
        )
        total += reference_dequantize_rowwise(
            row_codes, row_sf, ar[0], is_swizzled=False, fp8_max=EDEN_BLOCK_SCALE_MAX
        )
    mean = total / draws
    scale = target.abs().max()
    assert (mean - target).abs().mean() < 0.02 * scale


# --- wrapper layer, runs today ----------------------------------------------


@maybe_sm100
@torch.no_grad()
def test_register_fake_shapes_and_return_order():
    from torch._subclasses.fake_tensor import FakeTensorMode

    M, N, E = 512, 256, 2
    with FakeTensorMode():
        dy = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sv = torch.empty(128, dtype=torch.int8, device="cuda")
        offs = torch.empty(E, dtype=torch.int32, device="cuda")
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        rng = torch.empty(4, dtype=torch.int64, device="cuda")
        row, col = triton_group_row_rht_col_rht_amax(
            dy, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None
        )
        assert row.shape == (E,) and col.shape == (E,)
        out = triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, amax, amax, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, rng, None
        )
    # Columnwise pair first.
    assert [tuple(t.shape) for t in out] == [
        (N, M // 2),
        (N, M // 16),
        (M, N // 2),
        (M, N // 16),
    ]


@maybe_sm100
@pytest.mark.parametrize("which", ["dgrad_rht", "wgrad_rht"])
@torch.no_grad()
def test_rejects_a_non_128_sign_vector(which):
    M, N, E = 512, 256, 2
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    good = torch.ones(128, dtype=torch.int8, device="cuda")
    bad = torch.ones(16, dtype=torch.int8, device="cuda")
    d, w = (bad, good) if which == "dgrad_rht" else (good, bad)
    with pytest.raises(ValueError, match=rf"{which} must be a \(128,\) tensor"):
        triton_group_row_rht_col_rht_amax(
            dy, d, w, offs, E, M, N, VARYING_FIRST_DIM, offs[-1:]
        )


@maybe_sm100
@torch.no_grad()
def test_ms_eden_always_requires_an_rng_state():
    """Unlike the cast quantizers there is no deterministic mode to fall back to."""
    M, N, E = 512, 256, 2
    dy = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    sv = torch.ones(128, dtype=torch.int8, device="cuda")
    amax = torch.ones(E, dtype=torch.float32, device="cuda")
    with pytest.raises(TypeError, match="rng_state must be a torch.Tensor"):
        triton_group_row_rht_col_rht_quantize_ms_eden(
            dy, amax, amax, sv, sv, offs, E, M, N, VARYING_FIRST_DIM, None, offs[-1:]
        )
