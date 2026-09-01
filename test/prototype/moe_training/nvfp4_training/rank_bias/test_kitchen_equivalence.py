"""Optional cross-check against a real kitchen install.

Skipped unless ``kitchen`` imports, which needs kitchen built (``kitchen.ext``
is a compiled extension). This is the check the portable suite cannot do: it
compares the oracle in :mod:`nvfp4_reference` -- and therefore the CuTe DSL
kernels, which are bitwise identical to it -- against kitchen's own
``nvfp_utils.to_nvfp`` / ``from_nvfp`` for the recipe 6302 leaf path.

The stochastic path is compared by feeding kitchen's ``cast_to_fp4_e2m1_sr`` the
same Philox uniforms this package uses, so the two rounding decisions are driven
by identical random bits.

The last tests drive recipes 6302 and 9004 through the real quantize ops --
``QuantizeOpNVFP4Emulation`` and, for 9004, the ``QuantizeOpFP8HadamardTransform``
wrapper around it -- which dispatch to the psx ``clippy`` CUDA kernels; see
``test_psx_clippy_equivalence.py`` for those kernels on their own.
"""

from __future__ import annotations

import pytest
import torch

nvfp_utils = pytest.importorskip(
    "kitchen.nvfp_utils", reason="kitchen is not built in this environment"
)
from kitchen import quantization, utils  # noqa: E402

from torchao.prototype.moe_training.nvfp4_training.rank_bias import analyze_rank_bias  # noqa: E402
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_cutedsl as ct  # noqa: E402
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_reference as ref  # noqa: E402
from torchao.prototype.moe_training.nvfp4_training.rank_bias import rht  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

SEED = 21


def _qparams(use_sr: bool) -> "quantization.QParams":
    """``get_qlinear_params_from_qat_params(6302).g_params``, built directly.

    Building it through ``kitchen.config`` would construct
    ``QuantizeOpNVFP4Emulation``, whose ``__post_init__`` imports ``psx_formats``;
    the fields below are exactly what ``QuantizeRecipe.NVFP4_EMULATION`` sets for
    G, plus recipe 6302's ``use_sr``.
    """
    return quantization.QParams(
        scaling_type=quantization.ScalingType.PER_1D_BLOCK,
        quant_dtype=utils.Fp4Formats.E2M1,
        quant_tile_shape=(1, 16),
        pow_2_scales=False,
        tensor_type=quantization.TensorType.G,
        use_sr=use_sr,
    )


def _kitchen_qdq(x: torch.Tensor, *, transpose: bool, use_sr: bool) -> torch.Tensor:
    """kitchen's leaf QDQ: pad -> to_nvfp -> from_nvfp -> bf16 -> trim."""
    qparams = _qparams(use_sr)
    padded = ref.pad_2d(x, *(ref.PAD_TRANSPOSE if transpose else ref.PAD_IDENTITY))
    # The psx transpose kernel blocks 16 elements down each column, i.e. the
    # identity path applied to x.T.
    data = padded.t().contiguous() if transpose else padded
    block_descale, global_descale, data_q, _ = nvfp_utils.to_nvfp(
        data,
        qparams,
        known_amax=None,
        transposed=transpose,
        scale_rounding_mode=quantization.ClippyScaleRoundingMode.E4M3_RNE,
    )
    out = nvfp_utils.from_nvfp(
        block_descale, global_descale, data_q, qparams, is_qtranspose=transpose
    ).to(torch.bfloat16)
    if transpose:
        out = out.t()
    return out[: x.shape[0], : x.shape[1]]


@pytest.fixture
def philox_rand(monkeypatch):
    """Feed kitchen's SR cast this package's Philox stream instead of torch.rand."""
    calls = []

    def fake_rand(shape, device=None, **kwargs):
        calls.append(shape)
        numel = 1
        for dim in shape:
            numel *= dim
        return ref.philox_uniform(numel, SEED, device).reshape(shape)

    monkeypatch.setattr(torch, "rand", fake_rand)
    return calls


@pytest.mark.parametrize("shape", [(256, 512), (130, 48)])
@pytest.mark.parametrize("transpose", [False, True])
def test_reference_matches_kitchen_rne(shape, transpose):
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    want = _kitchen_qdq(x, transpose=transpose, use_sr=False)
    assert torch.equal(
        ref.quant_dequant(x, transpose=transpose).view(torch.int16),
        want.view(torch.int16),
    )
    assert torch.equal(
        ct.quant_dequant(x, transpose=transpose).contiguous().view(torch.int16),
        want.view(torch.int16),
    )


@pytest.mark.parametrize("shape", [(256, 512), (130, 48)])
@pytest.mark.parametrize("transpose", [False, True])
def test_reference_matches_kitchen_sr(shape, transpose, philox_rand):
    g = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    want = _kitchen_qdq(x, transpose=transpose, use_sr=True)
    assert len(philox_rand) == 1, "kitchen drew random bits more than once"
    got = ref.quant_dequant(x, transpose=transpose, use_sr=True, seed=SEED)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(
        ct.quant_dequant(x, transpose=transpose, use_sr=True, seed=SEED)
        .contiguous()
        .view(torch.int16),
        want.view(torch.int16),
    )


# ---------------------------------------------------------------------------
# The real dispatch path: recipe 6302 through QuantizeOpNVFP4Emulation, which
# calls the psx clippy CUDA kernels. RNE only (X / W); G rounds stochastically
# inside the kernel, so it has no bitwise comparison -- see
# test_psx_clippy_equivalence.py for what is checked there instead.
# ---------------------------------------------------------------------------


def _recipe_qdq(x: torch.Tensor, *, tensor_type, transpose: bool) -> torch.Tensor:
    """``quantize_tensor_with_recipe(x, 6302, tensor_type)``, identity or transpose."""
    from kitchen.config import get_qlinear_params_from_qat_params

    qlp = get_qlinear_params_from_qat_params(6302)
    qparams = {
        quantization.TensorType.X: qlp.x_params,
        quantization.TensorType.W: qlp.w_params,
    }[tensor_type]
    result = qlp.quantize_op.quantize(
        x=x,
        qparams=qparams,
        return_identity=True,
        return_transpose=True,
        reduce_amax=False,
        tp_group=None,
        input_meta=None,
        quant_stat_config=None,
        is_first_microbatch=False,
    )
    group = result.transpose if transpose else result.identity
    qdq, _ = qlp.quantize_op.dequantize(
        qtensor_group=group, qparams=qparams, is_data_t=transpose
    )
    return qdq[: x.shape[0], : x.shape[1]]


@pytest.mark.parametrize("shape", [(256, 512), (130, 48)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("tensor_type", ["X", "W"])
def test_matches_recipe_6302_dispatch_path(shape, transpose, tensor_type):
    g = torch.Generator(device="cuda").manual_seed(6)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    want = _recipe_qdq(
        x, tensor_type=getattr(quantization.TensorType, tensor_type), transpose=transpose
    )
    assert torch.equal(
        ref.quant_dequant(x, transpose=transpose).view(torch.int16),
        want.contiguous().view(torch.int16),
    )
    assert torch.equal(
        ct.quant_dequant(x, transpose=transpose).contiguous().view(torch.int16),
        want.contiguous().view(torch.int16),
    )


# ---------------------------------------------------------------------------
# Recipe 9004 (== 6304): the wgrad RHT on top of the 6302 leaf. Only the
# transpose lane of X and G is rotated; W is never rotated and G's identity lane
# is untouched, so those must stay bit-identical to 6302.
# ---------------------------------------------------------------------------


def _hadamard_op():
    """The 9004 op, with its Hadamard matrices materialized."""
    from kitchen.config import get_qlinear_params_from_qat_params

    qlp = get_qlinear_params_from_qat_params(9004)
    op = qlp.quantize_op
    op.initialize_hadamard_matrices(
        torch.zeros(16, 16, device="cuda", dtype=torch.bfloat16), qlp.g_params.tensor_type
    )
    return qlp, op


def _recipe_9004_qdq(x: torch.Tensor, *, tensor_type, transpose: bool) -> torch.Tensor:
    """Recipe 9004 end to end through QuantizeOpFP8HadamardTransform."""
    qlp, op = _hadamard_op()
    qparams = {
        quantization.TensorType.X: qlp.x_params,
        quantization.TensorType.G: qlp.g_params,
    }[tensor_type]
    result = op.quantize(
        x=x,
        qparams=qparams,
        return_identity=True,
        return_transpose=True,
        reduce_amax=False,
        tp_group=None,
        input_meta=None,
        quant_stat_config=None,
        is_first_microbatch=False,
    )
    group = result.transpose if transpose else result.identity
    qdq, _ = op.dequantize(
        qtensor_group=group, qparams=qparams, is_data_t=transpose
    )
    return qdq[: x.shape[0], : x.shape[1]].contiguous()


@pytest.mark.parametrize("shape", [(256, 512), (130, 48), (48, 96)])
@pytest.mark.parametrize("inverse", [False, True])
def test_rht_matches_kitchen(shape, inverse):
    """The rotation alone, forward and inverse, bit for bit.

    kitchen folds 1/sqrt(16) into the cuBLAS alpha (fp32, before the bf16
    rounding); rht.transform scales afterwards. 0.25 is a power of two, so the
    two agree -- this is the test that says so rather than the argument.
    """
    _, op = _hadamard_op()
    gemm_type = op.get_gemm_types_for_tensor(quantization.TensorType.G)[1]
    forward, inv = rht.transform_matrices(rht.sign_vector(torch.device("cuda")))
    assert torch.equal(
        op.hadamard_param_dict[gemm_type]["random_hadamard_matrix"], forward
    )
    assert torch.equal(
        op.hadamard_param_dict[gemm_type]["inverse_random_hadamard_matrix"], inv
    )

    g = torch.Generator(device="cuda").manual_seed(8)
    x = rht.pad_rows((torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16))
    want = op.perform_random_hadamard_transform_ref(
        x, gemm_type, True, inverse_transform=inverse
    )
    got = rht.transform(x, inv if inverse else forward)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


@pytest.mark.parametrize("shape", [(256, 512), (130, 48)])
@pytest.mark.parametrize("transpose", [False, True])
def test_matches_recipe_9004_dispatch_path(shape, transpose):
    """X and X.T under 9004: RNE, so the rotated lane is bitwise comparable."""
    g = torch.Generator(device="cuda").manual_seed(6)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    want = _recipe_9004_qdq(x, tensor_type=quantization.TensorType.X, transpose=transpose)
    matrices = rht.transform_matrices(rht.sign_vector(x.device)) if transpose else None
    for backend in (ref, ct):
        got = analyze_rank_bias.leaf_qdq(
            backend, x, transpose=transpose, use_sr=False, matrices=matrices, seed=0
        )
        assert torch.equal(got.contiguous().view(torch.int16), want.view(torch.int16))


def test_9004_rotates_exactly_the_lanes_the_recipe_table_claims():
    """Pin ``Recipe.rht_transpose``: transpose lane only, X and G only, never W.

    G cannot be compared bitwise here -- ``use_sr`` applies to both of its lanes,
    so even the unrotated identity lane is stochastic -- but which lanes rotate
    is a property of the op, and that is exactly what the recipe table encodes.
    """
    _, op = _hadamard_op()
    for tensor_type in (quantization.TensorType.X, quantization.TensorType.G):
        enable_identity, enable_transpose = op.get_rht_settings_for_tensor(tensor_type)[:2]
        assert not enable_identity and enable_transpose
        assert tensor_type.name in analyze_rank_bias.RECIPES["9004"].rht_transpose
    assert op.get_rht_settings_for_tensor(quantization.TensorType.W)[:2] == (False, False)
    assert "W" not in analyze_rank_bias.RECIPES["9004"].rht_transpose


def test_9004_transpose_lane_sr_shares_kitchen_scales():
    """G.T under 9004 rounds stochastically inside the psx kernel, so it has no
    bitwise comparison. The rotation is pinned bitwise by test_rht_matches_kitchen;
    what is left is the QDQ of the rotated tensor, checked here with the bracket
    technique from test_psx_clippy_equivalence.py -- every kitchen SR output must
    land on one of the two grid points this package would choose between."""
    clippy = pytest.importorskip("test_psx_clippy_equivalence")
    g = torch.Generator(device="cuda").manual_seed(9)
    x = (torch.randn((256, 512), generator=g, device="cuda") * 3).to(torch.bfloat16)
    forward, _ = rht.transform_matrices(rht.sign_vector(x.device))
    rotated = rht.transform(rht.pad_rows(x), forward)

    saved = ref.philox_uniform
    try:
        ref.philox_uniform = lambda numel, seed, device: torch.full(
            (numel,), 0.0, device=device
        )
        high = ref.quant_dequant(rotated, transpose=True, use_sr=True).float()
        ref.philox_uniform = lambda numel, seed, device: torch.full(
            (numel,), 1.0 - 2**-24, device=device
        )
        low = ref.quant_dequant(rotated, transpose=True, use_sr=True).float()
    finally:
        ref.philox_uniform = saved

    for _ in range(8):
        out = clippy.clippy_qdq(rotated, transpose=True, use_sr=True).float()
        assert bool(((out == high) | (out == low)).all())
