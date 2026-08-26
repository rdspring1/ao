# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Grouped rowwise-cast + columnwise-RHT amax and quantize. Design doc §11.8, §11.9.

V2's forward activation path: RHT-128 with a *dynamic* sign tensor, where V1 uses
RHT-16 with a static list. The semantics are otherwise those of the shipped
``triton_group_rht_amax`` / ``triton_group_rht_quantize_row_col``, so the shipped ops
serve as a second oracle wherever the Hadamard size is not what is under test.
"""

import pytest
import torch

from ._assertions import assert_codes_bitwise, assert_scales_bitwise
from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100
from .nvfp4_reference import (
    reference_group_row_cast_col_rht_amax,
    reference_group_row_cast_col_rht_quantize,
    reference_row_cast_col_rht_amax,
)

_AMAX_IMPLEMENTED = False
_QUANTIZE_IMPLEMENTED = False
_needs_amax = kernel_gate(_AMAX_IMPLEMENTED, "group_row_cast_col_rht_amax_triton.py")
_needs_quantize = kernel_gate(
    _AMAX_IMPLEMENTED and _QUANTIZE_IMPLEMENTED,
    "group_row_cast_col_rht_quantize_triton.py",
)

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.group_hadamard_utils import (
        VARYING_FIRST_DIM,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_col_rht_amax_triton import (
        triton_group_row_cast_col_rht_amax,
    )
    from torchao.prototype.moe_training.nvfp4_training.group_row_cast_col_rht_quantize_triton import (
        triton_group_row_cast_col_rht_quantize,
    )


def _signs(device="cuda", seed=0):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (128,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).to(device)


def _packed(group_sizes, hidden, *, seed=0):
    """Row-concatenated groups plus the cumulative row-end offsets they imply."""
    torch.manual_seed(seed)
    total = sum(group_sizes)
    A = torch.randn(total, hidden, device="cuda", dtype=torch.bfloat16)
    offsets = torch.cumsum(
        torch.tensor(group_sizes, dtype=torch.int32, device="cuda"),
        0,
        dtype=torch.int32,
    )
    return A, offsets


# --- §11.8 amax -------------------------------------------------------------


@_needs_amax
@torch.no_grad()
def test_amax_single_group_matches_the_linear_reference():
    A, offs = _packed([256], 512)
    sv = _signs()
    col, row = triton_group_row_cast_col_rht_amax(
        A, sv, offs, 1, 256, 512, VARYING_FIRST_DIM, offs[-1:]
    )
    ref_col, ref_row = reference_row_cast_col_rht_amax(A, sv)
    torch.testing.assert_close(col[0], ref_col, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(row[0], ref_row, atol=0, rtol=0)


@_needs_amax
@pytest.mark.parametrize("group_sizes", [[128, 128], [256, 128, 384, 128]])
@torch.no_grad()
def test_amax_multi_group_matches_the_reference(group_sizes):
    A, offs = _packed(group_sizes, 512)
    sv = _signs()
    E = len(group_sizes)
    col, row = triton_group_row_cast_col_rht_amax(
        A, sv, offs, E, A.shape[0], 512, VARYING_FIRST_DIM, offs[-1:]
    )
    ref_col, ref_row = reference_group_row_cast_col_rht_amax(A, sv, offs, E)
    torch.testing.assert_close(col, ref_col, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(row, ref_row, atol=0, rtol=0)


@_needs_amax
@torch.no_grad()
def test_amax_per_group_isolation():
    """Scale one group by 1000x; only that group's amaxes may move."""
    group_sizes = [128, 128, 128, 128]
    A, offs = _packed(group_sizes, 512)
    sv = _signs()
    E = len(group_sizes)
    base = triton_group_row_cast_col_rht_amax(
        A, sv, offs, E, A.shape[0], 512, VARYING_FIRST_DIM, offs[-1:]
    )
    A2 = A.clone()
    A2[128:256] *= 1000.0
    hot = triton_group_row_cast_col_rht_amax(
        A2, sv, offs, E, A2.shape[0], 512, VARYING_FIRST_DIM, offs[-1:]
    )
    for g in range(E):
        if g == 1:
            continue
        assert hot[0][g].item() == base[0][g].item(), f"group {g} col leaked"
        assert hot[1][g].item() == base[1][g].item(), f"group {g} row leaked"


@_needs_amax
@torch.no_grad()
def test_rowwise_amax_is_untouched_by_the_sign_vector():
    """The rowwise value is a plain amax; only the columnwise one is transformed."""
    A, offs = _packed([256], 512)
    a = triton_group_row_cast_col_rht_amax(
        A, _signs(seed=0), offs, 1, 256, 512, VARYING_FIRST_DIM, offs[-1:]
    )
    b = triton_group_row_cast_col_rht_amax(
        A, _signs(seed=1), offs, 1, 256, 512, VARYING_FIRST_DIM, offs[-1:]
    )
    assert a[1][0].item() == b[1][0].item(), "rowwise amax must be bitwise unchanged"
    assert a[0][0].item() != b[0][0].item(), "columnwise amax must change"


@_needs_amax
@torch.no_grad()
def test_padded_rows_are_excluded():
    """Rows at or beyond ``logical_packed_length`` are capacity, never data.

    They are left uninitialized by the dispatcher, so a kernel that reads them would
    fold garbage into a group's amax.
    """
    A, offs = _packed([128, 128], 512)
    capacity = torch.full((512, 512), float("inf"), device="cuda", dtype=torch.bfloat16)
    capacity[:256] = A
    got = triton_group_row_cast_col_rht_amax(
        capacity, _signs(), offs, 2, 512, 512, VARYING_FIRST_DIM, offs[-1:]
    )
    assert torch.isfinite(torch.stack(got)).all(), "spare capacity leaked into an amax"


# --- §11.9 quantize ---------------------------------------------------------


@_needs_quantize
@pytest.mark.parametrize("group_sizes", [[256], [128, 128], [256, 128, 384]])
@torch.no_grad()
def test_quantize_matches_the_reference_per_group(group_sizes):
    A, offs = _packed(group_sizes, 512)
    sv = _signs()
    E = len(group_sizes)
    M = A.shape[0]
    col_amax, row_amax = triton_group_row_cast_col_rht_amax(
        A, sv, offs, E, M, 512, VARYING_FIRST_DIM, offs[-1:]
    )
    row_codes, row_sf, col_codes, col_sf = triton_group_row_cast_col_rht_quantize(
        A,
        sv,
        offs,
        E,
        M,
        512,
        VARYING_FIRST_DIM,
        row_amax,
        col_amax,
        offs[-1:],
        False,
    )
    refs = reference_group_row_cast_col_rht_quantize(A, row_amax, col_amax, sv, offs, E)
    start = 0
    for g, size in enumerate(group_sizes):
        row_ref, _ = refs[g]
        assert_codes_bitwise(
            row_codes[start : start + size], row_ref.codes, f"row codes[{g}]"
        )
        start += size


@_needs_quantize
@torch.no_grad()
def test_rowwise_output_is_unchanged_by_the_sign_vector():
    """Only the columnwise path is rotated; changing signs must not perturb rowwise."""
    A, offs = _packed([256], 512)
    outs = []
    for seed in (0, 1):
        sv = _signs(seed=seed)
        ca, ra = triton_group_row_cast_col_rht_amax(
            A, sv, offs, 1, 256, 512, VARYING_FIRST_DIM, offs[-1:]
        )
        outs.append(
            triton_group_row_cast_col_rht_quantize(
                A,
                sv,
                offs,
                1,
                256,
                512,
                VARYING_FIRST_DIM,
                ra,
                ca,
                offs,
                False,
            )
        )
    assert_codes_bitwise(outs[0][0], outs[1][0], "rowwise codes")
    assert_scales_bitwise(outs[0][1], outs[1][1], "rowwise scales")
    assert not torch.equal(outs[0][2], outs[1][2]), "columnwise codes must change"


# --- wrapper layer, runs today ----------------------------------------------


@maybe_sm100
@torch.no_grad()
def test_register_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode

    M, N, E = 512, 256, 2
    with FakeTensorMode():
        A = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sv = torch.empty(128, dtype=torch.int8, device="cuda")
        offs = torch.empty(E, dtype=torch.int32, device="cuda")
        amax = torch.empty(E, dtype=torch.float32, device="cuda")
        col, row = triton_group_row_cast_col_rht_amax(
            A, sv, offs, E, M, N, VARYING_FIRST_DIM, None
        )
        assert col.shape == (E,) and row.shape == (E,)
        out = triton_group_row_cast_col_rht_quantize(
            A,
            sv,
            offs,
            E,
            M,
            N,
            VARYING_FIRST_DIM,
            amax,
            amax,
            None,
            False,
        )
    assert [tuple(t.shape) for t in out] == [
        (M, N // 2),
        (M, N // 16),
        (N, M // 2),
        (N, M // 16),
    ]


@maybe_sm100
@torch.no_grad()
def test_sign_vector_must_be_a_128_element_device_tensor():
    """A tuple is V1's interface. Passing one here must fail rather than silently
    resolve through the value-keyed cache, which for a resampled buffer would return
    a stale RHT matrix."""
    M, N, E = 512, 256, 2
    A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    offs = torch.tensor([256, 512], dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match=r"sign_vector must be a \(128,\) tensor"):
        triton_group_row_cast_col_rht_amax(
            A,
            torch.ones(16, dtype=torch.int8, device="cuda"),
            offs,
            E,
            M,
            N,
            VARYING_FIRST_DIM,
            offs[-1:],
        )
    with pytest.raises((TypeError, RuntimeError)):
        triton_group_row_cast_col_rht_amax(
            A, tuple(range(128)), offs, E, M, N, VARYING_FIRST_DIM, offs[-1:]
        )
