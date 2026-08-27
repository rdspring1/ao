# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Dense NVFP4 linear under the V2 and V1_REQUANT recipes. Design doc §15 and §16.

The load-bearing structural claim of these recipes is that a dense linear *is* the
degenerate one-group case of the grouped kernels. Several tests below exist to hold
that claim rather than to check numerics: that no non-grouped kernel is reachable,
and that the linear path and the grouped path agree at ``E = 1``.
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from torchao.prototype.moe_training.nvfp4_training import nvfp4_linear_v2 as v2_mod
from torchao.prototype.moe_training.nvfp4_training.nvfp4_recipe import NVFP4Recipe
from torchao.prototype.moe_training.nvfp4_training.nvfp4_rht_cadence import (
    resample_nvfp4_rht_signs,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import (
    NVFP4Linear,
    NVFP4TrainingConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import compute_error

from ._v2_marks import kernel_gate, kernel_skip, maybe_sm100

# Every grouped kernel these recipes call must be implemented before the numerics
# tests mean anything. V1_REQUANT needs only the three no-RHT ops; V2 needs all of them.
_V1_REQUANT_KERNELS_IMPLEMENTED = True
_V2_KERNELS_IMPLEMENTED = False
_needs_v1_requant = kernel_gate(
    _V1_REQUANT_KERNELS_IMPLEMENTED, "the §11.1/§11.6/§11.7 kernels"
)
_needs_v2 = kernel_gate(_V2_KERNELS_IMPLEMENTED, "the §11.1-§11.9 kernels")

# For tests that cover both recipes. Gating the whole test on V2 would keep the
# V1_REQUANT half unreachable through all of Phase A, which is exactly the half that
# has to hold before a V1_REQUANT convergence run.
_BOTH_RECIPES = [
    pytest.param(
        NVFP4Recipe.V1_REQUANT,
        marks=kernel_skip(
            _V1_REQUANT_KERNELS_IMPLEMENTED, "the §11.1/§11.6/§11.7 kernels"
        ),
        id="v1_requant",
    ),
    pytest.param(
        NVFP4Recipe.V2,
        marks=kernel_skip(_V2_KERNELS_IMPLEMENTED, "the §11.1-§11.9 kernels"),
        id="v2",
    ),
]

_M = _K = _N = 256

# Ops that belong to recipe V1 only. If any of these is dispatched while a
# V1_REQUANT or V2 layer runs, the "linear is grouped at num_tensors=1" design has
# been quietly abandoned somewhere.
_LINEAR_ONLY_OPS = {
    "torchao::triton_rht_amax",
    "torchao::triton_rht_quantize_row_col",
    "torchao::triton_weight_quantize_2d",
    "torchao::cutedsl_rht_amax",
    "torchao::cutedsl_rht_quantize_row_col",
    "torchao::cutedsl_weight_quantize_2d",
}


class _RecordOps(TorchDispatchMode):
    """Record every ``torchao::`` op dispatched inside the block."""

    def __init__(self):
        self.names = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name() if hasattr(func, "name") else str(func)
        if name.startswith("torchao::"):
            self.names.add(name.split(".")[0])
        return func(*args, **(kwargs or {}))


def _layer(recipe, *, seed=0):
    torch.manual_seed(seed)
    return NVFP4Linear(
        _K, _N, bias=False, device="cuda", dtype=torch.bfloat16, recipe=recipe
    )


def _inputs(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(_M, _K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return x


# ---------------------------------------------------------------------------
# Structural tests -- these run today
# ---------------------------------------------------------------------------


def test_the_v2_module_imports_no_linear_only_kernel():
    """Static half of the "no non-grouped op is reachable" claim.

    Cheap, runs without a GPU, and catches the mistake at review time rather than at
    dispatch time.
    """
    imported = {
        name
        for name in vars(v2_mod)
        if name.startswith(("triton_rht", "triton_weight", "cutedsl_"))
    }
    assert imported == set(), f"linear-only kernels imported: {sorted(imported)}"


def test_every_quantize_op_it_imports_is_grouped():
    """The positive form: everything it calls is a ``*_group_*`` op."""
    quantizers = {
        name
        for name in vars(v2_mod)
        if name.startswith("triton_") and not name.startswith("triton_group")
    }
    assert quantizers == set(), f"non-grouped kernels imported: {sorted(quantizers)}"


@maybe_sm100
def test_degenerate_group_offsets_are_cached_not_reallocated():
    """A fresh scalar tensor per forward is what makes torch.compile recompile every
    step, and under CUDA-graph capture it allocates into the graph pool."""
    a = v2_mod._degenerate_group_args(_M, "cuda:0")
    b = v2_mod._degenerate_group_args(_M, "cuda:0")
    assert a is b
    assert a.tolist() == [_M] and a.dtype == torch.int32


@maybe_sm100
def test_degenerate_group_offsets_are_not_cached_while_tracing():
    """The other half: a call made under tracing must NOT reach the cache.

    Under a FakeTensorMode this function returns a FakeTensor bound to that mode.
    Caching it makes lru_cache hand the same object to every later mode, which
    asserts "Mixing fake modes NYI" -- and full activation checkpointing retraces
    the checkpointed region under a second mode, so two modes see this function
    within one step. That is a compile-time crash no eager test can reach, and the
    test above passes either way, so it needs its own guard.

    is_compiling() is monkeypatched rather than driven through torch.compile because
    the contract under test is the guard itself; the real path is covered by the
    671B run.
    """
    v2_mod._degenerate_group_args_cached.cache_clear()
    with mock.patch.object(torch.compiler, "is_compiling", lambda: True):
        a = v2_mod._degenerate_group_args(_M, "cuda:0")
        b = v2_mod._degenerate_group_args(_M, "cuda:0")
    assert a is not b, "cache was consulted while tracing"
    assert v2_mod._degenerate_group_args_cached.cache_info().currsize == 0, (
        "a traced call populated the cache"
    )
    assert a.tolist() == [_M] and a.dtype == torch.int32


@maybe_sm100
@pytest.mark.parametrize("recipe", [NVFP4Recipe.V1_REQUANT, NVFP4Recipe.V2])
def test_shape_constraints_are_enforced(recipe):
    layer = _layer(recipe)
    with pytest.raises(ValueError, match="divisible by 128"):
        layer(torch.randn(100, _K, device="cuda", dtype=torch.bfloat16))


@maybe_sm100
@pytest.mark.parametrize("recipe", list(NVFP4Recipe))
def test_buffers_match_the_recipe(recipe):
    """V2 is the only recipe that draws a second sign vector, because it is the only
    one that rotates the dgrad axis."""
    layer = _layer(recipe)
    names = {n for n, _ in layer.named_buffers()}
    assert "_rht_sign_vector" in names and "_sr_seed" in names
    if recipe is NVFP4Recipe.V2:
        assert "_dgrad_rht_sign_vector" in names
        assert layer._rht_sign_vector.numel() == 128
    else:
        assert "_dgrad_rht_sign_vector" not in names
        assert layer._rht_sign_vector.numel() == 16


@maybe_sm100
def test_v1_default_is_unchanged():
    """An unchanged config must still build exactly the V1 layer it always did."""
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig())
    assert model[0].recipe is NVFP4Recipe.V1
    assert model[0]._rht_sign_vector.numel() == 16
    assert not hasattr(model[0], "_dgrad_rht_sign_vector")


@maybe_sm100
def test_v2_sign_buffers_resample_in_place():
    """§15's "same tensor identity, updated in place" requirement.

    The addresses must survive resampling or a CUDA graph captured around the step
    would replay against freed memory.
    """
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig(recipe=NVFP4Recipe.V2))
    layer = model[0]
    w_ptr = layer._rht_sign_vector.data_ptr()
    d_ptr = layer._dgrad_rht_sign_vector.data_ptr()

    updated = resample_nvfp4_rht_signs(model, seed=1, step=0, microbatch=0)
    assert updated == 2
    first_w = layer._rht_sign_vector.clone()
    first_d = layer._dgrad_rht_sign_vector.clone()

    resample_nvfp4_rht_signs(model, seed=1, step=0, microbatch=1)
    assert not torch.equal(first_w, layer._rht_sign_vector), (
        "wgrad resamples per microbatch"
    )
    assert torch.equal(first_d, layer._dgrad_rht_sign_vector), (
        "dgrad holds within a step"
    )

    resample_nvfp4_rht_signs(model, seed=1, step=1, microbatch=0)
    assert not torch.equal(first_d, layer._dgrad_rht_sign_vector), (
        "dgrad resamples per step"
    )

    assert layer._rht_sign_vector.data_ptr() == w_ptr
    assert layer._dgrad_rht_sign_vector.data_ptr() == d_ptr


@maybe_sm100
def test_static_recipes_are_never_resampled():
    """V1 and V1_REQUANT signs are fixed for the run; touching them would leak one
    cached RHT matrix per step through the value-keyed cache."""
    model = nn.Sequential(nn.Linear(_K, _N, bias=False)).cuda().bfloat16()
    quantize_(model, NVFP4TrainingConfig(recipe=NVFP4Recipe.V1_REQUANT))
    before = model[0]._rht_sign_vector.clone()
    assert resample_nvfp4_rht_signs(model, seed=1, step=5, microbatch=3) == 0
    assert torch.equal(before, model[0]._rht_sign_vector)


# ---------------------------------------------------------------------------
# Numerics -- gated on the kernel bodies
# ---------------------------------------------------------------------------


@_needs_v1_requant
@torch.no_grad()
def test_v1_requant_forward_vs_bf16():
    layer = _layer(NVFP4Recipe.V1_REQUANT)
    x = _inputs()
    got = layer(x)
    want = x @ layer.weight.t()
    assert compute_error(want.float(), got.float()) > 15.0


@_needs_v2
@torch.no_grad()
def test_v2_forward_vs_bf16():
    layer = _layer(NVFP4Recipe.V2)
    x = _inputs()
    got = layer(x)
    want = x @ layer.weight.t()
    assert compute_error(want.float(), got.float()) > 15.0


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
def test_gradients_vs_bf16_autograd(recipe):
    layer = _layer(recipe)
    x = _inputs()
    layer(x).sum().backward()

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = layer.weight.detach().clone().requires_grad_(True)
    (x_ref @ w_ref.t()).sum().backward()

    assert compute_error(x_ref.grad.float(), x.grad.float()) > 10.0
    assert compute_error(w_ref.grad.float(), layer.weight.grad.float()) > 10.0


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
def test_no_non_grouped_kernel_is_dispatched(recipe):
    """The runtime half of the "linear is grouped at num_tensors=1" claim.

    Every quantize and amax op that runs must be a grouped one. This is the test that
    fails if someone later "optimizes" the dense path onto a dedicated linear kernel
    without also keeping the two numerics in agreement.
    """
    layer = _layer(recipe)
    x = _inputs()
    with _RecordOps() as recorder:
        layer(x).sum().backward()
    assert recorder.names, "no torchao op was recorded; the probe is not working"
    leaked = recorder.names & _LINEAR_ONLY_OPS
    assert not leaked, f"non-grouped kernels dispatched: {sorted(leaked)}"


@maybe_sm100
@pytest.mark.parametrize("recipe", _BOTH_RECIPES)
@torch.no_grad()
def test_linear_matches_the_grouped_path_at_one_group(recipe):
    """The converse: driving the grouped entrypoint with ``offs = [M]`` must give the
    same answer as the dense one, bitwise."""
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
        nvfp4_v1_requant_grouped_mm,
        nvfp4_v2_grouped_mm,
    )

    layer = _layer(recipe)
    x = _inputs()
    dense = layer(x)
    offs = torch.tensor([_M], dtype=torch.int32, device="cuda")
    w3d = layer.weight.detach().unsqueeze(0)
    if recipe is NVFP4Recipe.V2:
        grouped = nvfp4_v2_grouped_mm(
            x.detach(),
            w3d,
            wgrad_rht=layer._rht_sign_vector,
            dgrad_rht=layer._dgrad_rht_sign_vector,
            sr_seed=layer._sr_seed,
            offs=offs,
        )
    else:
        grouped = nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w3d,
            sign_vector=layer.rht_sign_vector,
            sr_seed=layer._sr_seed,
            offs=offs,
        )
    torch.testing.assert_close(dense, grouped, atol=0, rtol=0)


@_needs_v2
def test_saved_tensors_hold_no_bf16_activation_or_weight_transpose():
    """§15 invariant 7: forward saves packed codes and scales only.

    Checked by size: anything the size of a bf16 activation or weight would show up
    immediately, since FP4 codes are a quarter of bf16 and the scales a sixteenth.
    """
    layer = _layer(NVFP4Recipe.V2)
    x = _inputs()
    out = layer(x)
    saved = out.grad_fn.saved_tensors
    for t in saved:
        assert t.dtype in (
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float32,
            torch.int8,
            torch.int64,
        ), f"unexpected saved dtype {t.dtype}"
    total = sum(t.numel() * t.element_size() for t in saved)
    bf16_activation = x.numel() * 2
    bf16_weight = layer.weight.numel() * 2
    assert total < bf16_activation + bf16_weight, (
        f"saved {total} bytes; a bf16 activation plus weight would be "
        f"{bf16_activation + bf16_weight}"
    )


@_needs_v2
def test_v2_backward_is_reproducible_for_a_fixed_rng_state(monkeypatch):
    """MS-EDEN draws fresh offsets per backward, so reproducibility has to be pinned
    by fixing the state the op receives."""
    fixed = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device="cuda")
    monkeypatch.setattr(v2_mod, "_backward_rng_state", lambda sr_seed: fixed)

    grads = []
    for _ in range(2):
        layer = _layer(NVFP4Recipe.V2)
        x = _inputs()
        layer(x).sum().backward()
        grads.append((x.grad.clone(), layer.weight.grad.clone()))
    assert torch.equal(grads[0][0], grads[1][0])
    assert torch.equal(grads[0][1], grads[1][1])
