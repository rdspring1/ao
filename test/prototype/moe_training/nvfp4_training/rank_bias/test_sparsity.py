"""Tests for :mod:`analyze_sparsity`. Portable: no kitchen, no psx_formats.

The sparsity numbers are only worth anything if ``flush`` really is "nonzero in,
FP4 zero out". These check that against cases whose answer is fixed by
construction, and against the independent analytic threshold.
"""

from __future__ import annotations

import pytest
import torch

from torchao.prototype.moe_training.nvfp4_training.rank_bias import analyze_sparsity as sp
from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_cutedsl
from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_reference
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_cutedsl
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

BACKENDS = [nvfp4_reference, nvfp4_cutedsl]


def _stats(x, backend=nvfp4_reference, transpose=False):
    return sp.sparsity_stats(
        x, transpose=transpose, backend=backend, matrices=None, rht_dim=16
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_flush_is_exact_on_a_hand_built_block(backend):
    """One large value and fifteen far below the FP4 threshold: 15/16 must flush.

    The block amax is 1.0, so the FP4 zero threshold is |x| < 1/24 ~= 0.0417;
    1e-3 is well under it and 1.0 is the amax itself. Nothing here is exactly
    zero, so every flushed element is information the format discarded.
    """
    block = torch.full((1, 16), 1e-3, device="cuda", dtype=torch.bfloat16)
    block[0, 0] = 1.0
    x = block.repeat(32, 1)
    stats = _stats(x, backend)
    assert stats.exact_zero == 0.0
    assert stats.flush == pytest.approx(15 / 16)
    assert stats.fp4_zero == pytest.approx(15 / 16)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exact_zeros_are_not_counted_as_flushed(backend):
    """A zero in the dump costs nothing; only nonzero-in / zero-out is a loss.

    This is the distinction the whole script turns on: a ReLU-family gradient is
    half exact zeros, and reporting those as "flushed" would say the format is
    destroying information it is in fact representing exactly.
    """
    block = torch.zeros((1, 16), device="cuda", dtype=torch.bfloat16)
    block[0, 0] = 1.0
    block[0, 1] = 1e-3  # the only genuinely flushed element
    x = block.repeat(32, 1)
    stats = _stats(x, backend)
    assert stats.exact_zero == pytest.approx(14 / 16)
    assert stats.flush == pytest.approx(1 / 16)
    assert stats.fp4_zero == pytest.approx(15 / 16)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("transpose", [False, True])
def test_flush_agrees_with_the_analytic_threshold(backend, transpose):
    """The measured flush and ``|x| / block_amax < 1/24`` are computed by two
    independent routes -- the real quantizer versus a ratio -- so agreement is a
    genuine cross-check rather than a tautology."""
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(512, 512, generator=g, device="cuda") * 3).to(torch.bfloat16)
    stats = _stats(x, backend, transpose=transpose)
    assert stats.flush == pytest.approx(stats.below_threshold, abs=2e-3)


def test_backends_agree():
    """The CuTe DSL and torch quantizers are bitwise identical, so every
    statistic derived from them must match exactly."""
    g = torch.Generator(device="cuda").manual_seed(1)
    x = (torch.randn(256, 256, generator=g, device="cuda") * 3).to(torch.bfloat16)
    for transpose in (False, True):
        a = _stats(x, nvfp4_reference, transpose)
        b = _stats(x, nvfp4_cutedsl, transpose)
        assert a == b, transpose


def test_all_zero_tensor_is_all_zeros_and_nothing_flushed():
    """amax 0 makes every scale degenerate; nothing may be NaN and nothing may
    be reported as a loss, because there was nothing to lose."""
    x = torch.zeros(64, 64, device="cuda", dtype=torch.bfloat16)
    stats = _stats(x)
    assert stats.exact_zero == 1.0
    assert stats.flush == 0.0
    assert stats.dead_block == 0.0  # no *live* block to be dead
    assert stats.p50_rel != stats.p50_rel  # nan: no nonzero element to rank


def test_rotation_destroys_structural_sparsity():
    """A Hadamard mixes 16 elements per output, so exact zeros do not survive it.

    This is why sparsity is reported per lane: the same tensor is 50% zeros on an
    unrotated lane and ~0% on a rotated one, and quoting one number for both
    would be wrong for whichever lane was not measured.
    """
    import rht

    g = torch.Generator(device="cuda").manual_seed(2)
    x = (torch.randn(256, 256, generator=g, device="cuda")).to(torch.bfloat16)
    x = torch.where(x > 0, x, torch.zeros_like(x))  # ~50% exact zeros
    matrices = rht.transform_matrices(rht.sign_vector(x.device, dim=16, lane="wgrad"))

    plain = _stats(x, transpose=True)
    rotated = sp.sparsity_stats(
        x, transpose=True, backend=nvfp4_reference, matrices=matrices, rht_dim=16
    )
    assert plain.exact_zero > 0.4
    assert rotated.raw_exact_zero == plain.exact_zero  # the model property is kept
    assert rotated.exact_zero < 0.01  # but the quantizer no longer sees it


@pytest.mark.parametrize("activation", ["swiglu", "relu2", "gelu", "geglu", "silu"])
def test_synthetic_grads_have_the_expected_layers_and_zero_structure(activation):
    """Gated activations produce an fc3 and no exact zeros; ReLU-family
    activations produce exact zeros in fc1 and only there.

    ``relu2`` differentiates to ``2*relu(h1)``, which is exactly zero on half its
    domain, so the ~50% is a structural prediction rather than an observation.
    """
    grads = sp.synthetic_mlp_grads(
        activation, tokens=512, hidden=256, ffn=512, seed=0, device=torch.device("cuda")
    )
    gated = activation in ("swiglu", "geglu")
    assert ("fc3" in grads) == gated
    assert set(grads) == ({"fc1", "fc2", "fc3"} if gated else {"fc1", "fc2"})
    # fc2's gradient is the incoming dy, untouched by the activation.
    assert (grads["fc2"] == 0).float().mean() == 0.0
    fc1_zeros = (grads["fc1"] == 0).float().mean().item()
    if activation == "relu2":
        assert 0.45 < fc1_zeros < 0.55
    else:
        assert fc1_zeros < 0.01


def test_eden_backend_reports_through_the_scale_keyword():
    """``_sr_kwarg`` must route to each quantizer's own SR spelling."""
    assert sp._sr_kwarg(nvfp4_reference) == "use_sr"
    assert sp._sr_kwarg(nvfp4_cutedsl) == "use_sr"
    assert sp._sr_kwarg(eden_reference) == "stochastic_round_scale"
    assert sp._sr_kwarg(eden_cutedsl) == "stochastic_round_scale"
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(256, 256, generator=g, device="cuda") * 3).to(torch.bfloat16)
    a = _stats(x, eden_reference)
    b = _stats(x, eden_cutedsl)
    assert a == b


# ---------------------------------------------------------------------------
# The real-dump path: shapes and dtypes that only occur in actual dumps
# ---------------------------------------------------------------------------

# MoE expert dumps carry however many tokens the router sent to that expert, so
# their row count is arbitrary; 1373 % 16 == 13 and 1373 % 128 == 93, which
# forces both the quantizer's padding and the RHT's.
RAGGED = [(1373, 1408), (2795, 2048), (17, 16), (4096, 576)]


@pytest.mark.parametrize("shape", RAGGED)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("rotate", [False, True])
def test_ragged_shapes_count_only_real_elements(shape, transpose, rotate):
    """Padding must not leak into any statistic.

    The quantizer pads to (32,16) / (128,64) and the RHT pads to a multiple of
    its dimension, all with zeros. If the validity mask were wrong those zeros
    would show up as exact zeros and drag ``flush`` toward nonsense, so checking
    ``numel`` and ``exact_zero`` together pins the mask from both sides.
    """
    import rht

    rows, cols = shape
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(rows, cols, generator=g, device="cuda") * 3).to(torch.bfloat16)
    matrices = (
        rht.transform_matrices(rht.sign_vector(x.device, dim=16, lane="wgrad"))
        if rotate
        else None
    )
    stats = sp.sparsity_stats(
        x, transpose=transpose, backend=nvfp4_reference, matrices=matrices, rht_dim=16
    )
    assert stats.numel == rows * cols  # the padding check: masked out, not counted
    if rotate:
        # The rotation sums 16 bf16 values and rounds back to bf16, so a few
        # cancel to exactly zero (~74 in 1.9M). Genuine, and orders of magnitude
        # below the >=0.7% that leaked padding rows would contribute.
        assert stats.exact_zero < 1e-3
    else:
        # Unrotated, the quantizer sees precisely the tensor that was passed in,
        # so the two zero counts must agree exactly. This is sharper than any
        # absolute bound and, unlike one, says nothing about the RNG: a Gaussian
        # draw may legitimately contain an exact zero, and both sides must then
        # report it.
        assert stats.exact_zero == stats.raw_exact_zero
    assert 0.0 < stats.flush < 0.5


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_dump_dtypes_are_accepted_without_a_downcast(dtype):
    """Dumps arrive at their own dtype and are handed over unchanged.

    Rounding an fp32 dump to bfloat16 before measuring would change the very
    quantity being measured, so the script must not do it and both quantizers
    must accept fp32 directly.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    x = (torch.randn(256, 256, generator=g, device="cuda") * 3).to(dtype)
    for backend in BACKENDS:
        stats = _stats(x, backend)
        assert stats.numel == 256 * 256
        assert 0.0 < stats.flush < 0.5


def test_three_dimensional_dumps_are_flattened_like_the_sweep_does():
    """A [seq, batch, hidden] dump collapses to 2-D on the last axis, which is
    what analyze_rank_bias.flatten_to_2d does and what the block axis assumes."""
    from analyze_rank_bias import flatten_to_2d

    g = torch.Generator(device="cuda").manual_seed(0)
    x3 = (torch.randn(8, 512, 256, generator=g, device="cuda")).to(torch.bfloat16)
    flat = flatten_to_2d(x3)
    assert flat.shape == (8 * 512, 256)
    assert _stats(flat).numel == 8 * 512 * 256
