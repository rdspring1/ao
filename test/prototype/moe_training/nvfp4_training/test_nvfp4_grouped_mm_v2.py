# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 grouped GEMMs and MoE FFN recipe routing. Design doc §17.

§17 routes FC1 (``w1``/``w3``) to V1_REQUANT and FC2 (``w2``) to V2. The tests that
matter most here are the routing ones: that each layer reaches only its own recipe's
kernels, and that FC1 and FC2 own independent sign vectors and seeds. A shared buffer
between them is an explicit test failure, not a performance detail.
"""

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from torchao.quantization.utils import compute_error

from ._v2_marks import TRITON_AVAILABLE, kernel_gate, maybe_sm100

_V1_REQUANT_KERNELS_IMPLEMENTED = True
_V2_KERNELS_IMPLEMENTED = False
_needs_v1_requant = kernel_gate(
    _V1_REQUANT_KERNELS_IMPLEMENTED, "the §11.1/§11.6/§11.7 kernels"
)
_needs_v2 = kernel_gate(_V2_KERNELS_IMPLEMENTED, "the §11.1-§11.9 kernels")

if TRITON_AVAILABLE:
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm_v2 import (
        nvfp4_v1_requant_grouped_mm,
        nvfp4_v2_grouped_mm,
    )

_MS_EDEN_OP = "torchao::triton_group_row_rht_col_rht_quantize_ms_eden"
_SR_CAST_OP = "torchao::triton_group_rht_quantize_row_col"
_ROTATED_REQUANT_OP = "torchao::triton_group_col_rht_requantize"
_PLAIN_REQUANT_OP = "torchao::triton_group_col_cast_requantize"


class _RecordOps(TorchDispatchMode):
    def __init__(self):
        self.names = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name() if hasattr(func, "name") else str(func)
        if name.startswith("torchao::"):
            self.names.add(name.split(".")[0])
        return func(*args, **(kwargs or {}))


def _signs(seed=0, n=128):
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(0, 2, (n,), generator=generator, dtype=torch.int8)
    return (bits * 2 - 1).cuda()


def _seed(value):
    return torch.tensor([value], dtype=torch.int64, device="cuda")


def _moe_inputs(group_sizes, D, F_dim, *, seed=0):
    torch.manual_seed(seed)
    E = len(group_sizes)
    x = torch.randn(
        sum(group_sizes), D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    kw = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w1 = (torch.randn(E, F_dim, D, **kw) * 0.02).detach().requires_grad_(True)
    w3 = (torch.randn(E, F_dim, D, **kw) * 0.02).detach().requires_grad_(True)
    w2 = (torch.randn(E, D, F_dim, **kw) * 0.02).detach().requires_grad_(True)
    offs = torch.cumsum(
        torch.tensor(group_sizes, dtype=torch.int32, device="cuda"),
        0,
        dtype=torch.int32,
    )
    return x, w1, w2, w3, offs


def _ffn(x, w1, w2, w3, offs, state):
    """§17's call shape: FC1 on V1_REQUANT, FC2 on V2, one ``offs`` per forward."""
    gate = nvfp4_v1_requant_grouped_mm(
        x, w1, sign_vector=state["fc1_signs"], sr_seed=state["fc1_seed"], offs=offs
    )
    up = nvfp4_v1_requant_grouped_mm(
        x, w3, sign_vector=state["fc1_signs"], sr_seed=state["fc1_seed"], offs=offs
    )
    hidden = F.silu(gate) * up
    return nvfp4_v2_grouped_mm(
        hidden,
        w2,
        wgrad_rht=state["fc2_wgrad"],
        dgrad_rht=state["fc2_dgrad"],
        sr_seed=state["fc2_seed"],
        offs=offs,
    )


def _state():
    return {
        "fc1_signs": tuple(int(v) for v in _signs(seed=10, n=16).tolist()),
        "fc1_seed": _seed(11),
        "fc2_wgrad": _signs(seed=20),
        "fc2_dgrad": _signs(seed=21),
        "fc2_seed": _seed(22),
    }


def _bf16_ffn(x, w1, w2, w3, group_sizes):
    out, start = [], 0
    for e, size in enumerate(group_sizes):
        xs = x[start : start + size]
        h = F.silu(xs @ w1[e].t()) * (xs @ w3[e].t())
        out.append(h @ w2[e].t())
        start += size
    return torch.cat(out)


# ---------------------------------------------------------------------------
# Numerics and routing -- gated on the kernel bodies
# ---------------------------------------------------------------------------


@_needs_v1_requant
@torch.no_grad()
def test_fc1_alone_vs_bf16_grouped_reference():
    sizes = [128, 128]
    x, w1, _, _, offs = _moe_inputs(sizes, 256, 512)
    got = nvfp4_v1_requant_grouped_mm(
        x.detach(), w1.detach(), sign_vector=(1,) * 16, sr_seed=_seed(1), offs=offs
    )
    want = torch.cat(
        [
            x[s : s + n].detach() @ w1[e].detach().t()
            for e, (s, n) in enumerate(zip([0, 128], sizes))
        ]
    )
    assert compute_error(want.float(), got.float()) > 12.0


@_needs_v2
@torch.no_grad()
def test_fc2_alone_vs_bf16_grouped_reference():
    sizes = [128, 128]
    x, _, w2, _, offs = _moe_inputs(sizes, 512, 256)
    h = torch.randn(sum(sizes), 256, device="cuda", dtype=torch.bfloat16)
    got = nvfp4_v2_grouped_mm(
        h,
        w2.detach(),
        wgrad_rht=_signs(0),
        dgrad_rht=_signs(1),
        sr_seed=_seed(1),
        offs=offs,
    )
    want = torch.cat(
        [
            h[s : s + n] @ w2[e].detach().t()
            for e, (s, n) in enumerate(zip([0, 128], sizes))
        ]
    )
    assert compute_error(want.float(), got.float()) > 12.0


@_needs_v2
def test_full_ffn_forward_and_backward():
    sizes = [128, 256, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    out = _ffn(x, w1, w2, w3, offs, _state())
    assert out.shape == (sum(sizes), 256)
    out.float().square().mean().backward()
    for name, p in (("x", x), ("w1", w1), ("w2", w2), ("w3", w3)):
        assert p.grad is not None, f"no gradient reached {name}"
        assert torch.isfinite(p.grad).all(), f"{name} gradient has non-finite values"


@_needs_v2
def test_recipe_routing_reaches_only_its_own_kernels():
    """§17 smoke test 4: FC1 calls no MS-EDEN op; FC2 calls no SR-cast op."""
    sizes = [128, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    state = _state()

    with _RecordOps() as fc1:
        nvfp4_v1_requant_grouped_mm(
            x, w1, sign_vector=state["fc1_signs"], sr_seed=state["fc1_seed"], offs=offs
        ).sum().backward()
    assert _MS_EDEN_OP not in fc1.names, "V1_REQUANT must not reach MS-EDEN"
    assert _ROTATED_REQUANT_OP not in fc1.names, "V1_REQUANT applies no dgrad rotation"
    assert _PLAIN_REQUANT_OP in fc1.names

    h = torch.randn(
        sum(sizes), 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    with _RecordOps() as fc2:
        nvfp4_v2_grouped_mm(
            h,
            w2,
            wgrad_rht=state["fc2_wgrad"],
            dgrad_rht=state["fc2_dgrad"],
            sr_seed=state["fc2_seed"],
            offs=offs,
        ).sum().backward()
    assert _SR_CAST_OP not in fc2.names, "V2 must not reach the SR cast quantizer"
    assert _PLAIN_REQUANT_OP not in fc2.names, "V2 requantizes with a rotation"
    assert _MS_EDEN_OP in fc2.names and _ROTATED_REQUANT_OP in fc2.names


@_needs_v2
def test_recipes_can_be_swapped():
    """§17 extra test: the FC1/FC2 split is a configuration choice, not an assumption.

    Running FC1 on V2 and FC2 on V1_REQUANT must work. If it does not, something has
    hard-coded the routing that §17 describes as a tunable.
    """
    sizes = [128, 128]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    gate = nvfp4_v2_grouped_mm(
        x, w1, wgrad_rht=_signs(0), dgrad_rht=_signs(1), sr_seed=_seed(1), offs=offs
    )
    out = nvfp4_v1_requant_grouped_mm(
        F.silu(gate),
        w2.transpose(-2, -1).contiguous(),
        sign_vector=(1,) * 16,
        sr_seed=_seed(2),
        offs=offs,
    )
    out.sum().backward()
    assert torch.isfinite(x.grad).all()


@_needs_v2
@torch.no_grad()
def test_uneven_groups_match_the_bf16_reference_per_group():
    sizes = [128, 384, 256]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    got = _ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), offs, _state())
    want = _bf16_ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), sizes)
    assert compute_error(want.float(), got.float()) > 8.0


@_needs_v2
def test_single_expert_matches_three_dense_linears():
    """§17 smoke test 5: at ``E = 1`` the FFN reduces to two linears plus one."""
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear_v2 import (
        nvfp4_linear_v1_requant,
        nvfp4_linear_v2,
    )

    sizes = [256]
    x, w1, w2, w3, offs = _moe_inputs(sizes, 256, 512)
    state = _state()
    grouped = _ffn(x.detach(), w1.detach(), w2.detach(), w3.detach(), offs, state)

    gate = nvfp4_linear_v1_requant(
        x.detach(),
        w1.detach()[0],
        sign_vector=state["fc1_signs"],
        sr_seed=state["fc1_seed"],
    )
    up = nvfp4_linear_v1_requant(
        x.detach(),
        w3.detach()[0],
        sign_vector=state["fc1_signs"],
        sr_seed=state["fc1_seed"],
    )
    dense = nvfp4_linear_v2(
        F.silu(gate) * up,
        w2.detach()[0],
        wgrad_rht=state["fc2_wgrad"],
        dgrad_rht=state["fc2_dgrad"],
        sr_seed=state["fc2_seed"],
    )
    torch.testing.assert_close(grouped, dense, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Wrapper layer -- runs today
# ---------------------------------------------------------------------------


@maybe_sm100
def test_fc1_and_fc2_state_must_be_independent():
    """§17: once the recipes differ, FC1 and FC2 can no longer share a sign vector or
    a seed. Pinned as a property of the state the caller assembles."""
    state = _state()
    assert state["fc1_seed"].item() != state["fc2_seed"].item()
    assert len(state["fc1_signs"]) == 16, "V1_REQUANT is RHT-16"
    assert state["fc2_wgrad"].numel() == 128, "V2 is RHT-128"
    assert not torch.equal(state["fc2_wgrad"], state["fc2_dgrad"])
    assert state["fc2_wgrad"].data_ptr() != state["fc2_dgrad"].data_ptr()


@maybe_sm100
@torch.no_grad()
def test_offs_is_required_and_validated():
    x, w1, _, _, offs = _moe_inputs([128, 128], 256, 512)
    with pytest.raises(ValueError, match="offs is required"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(), w1.detach(), sign_vector=(1,) * 16, sr_seed=_seed(1), offs=None
        )
    with pytest.raises(ValueError, match="1D int32"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w1.detach(),
            sign_vector=(1,) * 16,
            sr_seed=_seed(1),
            offs=offs.long(),
        )
    with pytest.raises(ValueError, match="one group-end offset per expert"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(),
            w1.detach(),
            sign_vector=(1,) * 16,
            sr_seed=_seed(1),
            offs=offs[:1],
        )


@maybe_sm100
@torch.no_grad()
def test_v2_requires_128_element_sign_tensors():
    x, _, w2, _, offs = _moe_inputs([128, 128], 512, 256)
    h = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"wgrad_rht must be a \(128,\) tensor"):
        nvfp4_v2_grouped_mm(
            h,
            w2.detach(),
            wgrad_rht=_signs(0, n=16),
            dgrad_rht=_signs(1),
            sr_seed=_seed(1),
            offs=offs,
        )


@maybe_sm100
@torch.no_grad()
def test_contraction_dimension_mismatch_is_caught():
    x, w1, _, _, offs = _moe_inputs([128, 128], 256, 512)
    bad = torch.randn(2, 512, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contraction dimensions differ"):
        nvfp4_v1_requant_grouped_mm(
            x.detach(), bad, sign_vector=(1,) * 16, sr_seed=_seed(1), offs=offs
        )
