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
from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_cutedsl as eden_ct  # noqa: E402
from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_reference as eden_ref  # noqa: E402
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
            backend,
            x,
            transpose=transpose,
            use_sr=False,
            matrices=matrices,
            rht_dim=16,
            seed=0,
        )
        assert torch.equal(got.contiguous().view(torch.int16), want.view(torch.int16))


def test_9004_rotates_exactly_the_lanes_the_recipe_table_claims():
    """Pin ``Recipe.rht_gemms``: transpose lane only, X and G only, never W.

    G cannot be compared bitwise here -- ``use_sr`` applies to both of its lanes,
    so even the unrotated identity lane is stochastic -- but which lanes rotate
    is a property of the op, and that is exactly what the recipe table encodes.
    """
    _, op = _hadamard_op()
    recipe = analyze_rank_bias.RECIPES["9004"]
    assert recipe.rht_dim == 16
    for tensor_type in quantization.TensorType:
        if tensor_type.name not in ("X", "W", "G"):
            continue
        enable_identity, enable_transpose = op.get_rht_settings_for_tensor(
            tensor_type
        )[:2]
        for transpose, enabled in ((False, enable_identity), (True, enable_transpose)):
            gemm = analyze_rank_bias.GEMM_TYPES[(tensor_type.name, transpose)]
            assert (gemm in recipe.rht_gemms) == enabled, (tensor_type, transpose)
    # The concrete claim the table makes, spelled out so a vacuous loop is visible.
    assert recipe.rht_gemms == frozenset({"wgrad"})


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


# ---------------------------------------------------------------------------
# Recipe 100483 (V2): MS-EDEN on G / G.T, RHT-128 on both backward GEMMs
# ---------------------------------------------------------------------------


def _eden_op(seed: int):
    """``QuantizeOpEdenFP4Emulation`` with a pinned SR seed.

    Recipe 100483 sets ``iterate_rng_seed`` and ``mix_rank_in_seed``, so a real
    run draws the seed from the ambient CPU generator and folds the rank in
    (``_next_rng_seed``). Both are switched off here so the kernel consumes a
    seed this test chooses -- which is the whole reason the Eden path can be
    compared bitwise while the 9004 SR path cannot.
    """
    from kitchen import quantization_eden_fp4

    return quantization_eden_fp4.QuantizeOpEdenFP4Emulation(
        rng_seed=seed,
        iterate_rng_seed=False,
        mix_rank_in_seed=False,
        stochastic_round_scale=True,
    )


def _eden_qparams() -> "quantization.QParams":
    """``get_qlinear_params_from_qat_params(100483).g_params``, built directly.

    ``_create_rht_technique_recipe`` sets ``enable_eden_scaling`` on ``g_params``
    and leaves the E2M1 / (1,16) / PER_1D_BLOCK fields at the NVFP4_EMULATION
    defaults; ``test_recipe_100483_g_params_match_the_hand_built_ones`` pins that
    against the real config rather than trusting this.
    """
    return quantization.QParams(
        scaling_type=quantization.ScalingType.PER_1D_BLOCK,
        quant_dtype=utils.Fp4Formats.E2M1,
        quant_tile_shape=(1, 16),
        pow_2_scales=False,
        tensor_type=quantization.TensorType.G,
        enable_eden_scaling=True,
    )


def _kitchen_eden_qdq(x: torch.Tensor, *, transpose: bool, seed: int) -> torch.Tensor:
    """kitchen's Eden leaf QDQ. The op does not pad and dequantize is a no-op cast."""
    op = _eden_op(seed)
    qparams = _eden_qparams()
    result = op.quantize(
        x,
        qparams,
        return_identity=not transpose,
        return_transpose=transpose,
        reduce_amax=False,
    )
    group = result.transpose if transpose else result.identity
    qdq, _ = op.dequantize(qtensor_group=group, qparams=qparams, is_data_t=transpose)
    return qdq.contiguous()


def test_recipe_100483_g_params_match_the_hand_built_ones():
    """The real recipe's G params are what ``_eden_qparams`` claims, and its G
    quantizer is the Eden op with the two seed-mixing flags 100483 sets."""
    from kitchen.config import get_qlinear_params_from_qat_params

    qlp = get_qlinear_params_from_qat_params(100483)
    built = _eden_qparams()
    for field in ("scaling_type", "quant_dtype", "quant_tile_shape", "pow_2_scales"):
        assert getattr(qlp.g_params, field) == getattr(built, field), field
    assert qlp.g_params.enable_eden_scaling
    assert not qlp.x_params.enable_eden_scaling
    assert not qlp.w_params.enable_eden_scaling

    g_quantizer = qlp.quantize_op.base_quantize_op.g_quantizer
    assert type(g_quantizer).__name__ == "QuantizeOpEdenFP4Emulation"
    assert g_quantizer.stochastic_round_scale
    assert g_quantizer.iterate_rng_seed and g_quantizer.mix_rank_in_seed
    assert g_quantizer.correction_dim == 16 and not g_quantizer.ue5m3_scale
    # X and W keep the psx NVFP4 quantizer, which is what RECIPES encodes.
    assert (
        type(qlp.quantize_op.base_quantize_op.x_quantizer).__name__
        == "QuantizeOpNVFP4Emulation"
    )
    assert analyze_rank_bias.RECIPES["100483"].quantizer == {
        "X": "nvfp4",
        "W": "nvfp4",
        "G": "eden",
    }


@pytest.mark.parametrize("shape", [(256, 512), (128, 128), (384, 256)])
@pytest.mark.parametrize("transpose", [False, True])
def test_eden_matches_kitchen_bitwise(shape, transpose):
    """The MS-EDEN leaf, bit for bit, against the real kernel -- SR included.

    This is the claim the 9004 path could never make: the data is RNE and the one
    random step is a software bit-twiddle on a Philox word whose seed is a kernel
    argument, so both sides round on identical bits without monkeypatching
    anything.
    """
    g = torch.Generator(device="cuda").manual_seed(4)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    want = _kitchen_eden_qdq(x, transpose=transpose, seed=SEED)
    for backend in (eden_ref, eden_ct):
        got = backend.quant_dequant(x, transpose=transpose, use_sr=True, seed=SEED)
        assert torch.equal(
            got.contiguous().view(torch.int16), want.view(torch.int16)
        ), backend.__name__


@pytest.mark.parametrize("transpose", [False, True])
def test_eden_matches_kitchen_with_rne_scales(transpose):
    """The deterministic branch too, which isolates the correction from the RNG.

    If only the SR test existed, a correction bug and a compensating RNG bug
    could cancel; this one has no randomness to hide in.
    """
    g = torch.Generator(device="cuda").manual_seed(5)
    x = (torch.randn((256, 256), generator=g, device="cuda") * 3).to(torch.bfloat16)
    from kitchen import quantization_eden_fp4

    op = quantization_eden_fp4.QuantizeOpEdenFP4Emulation(
        rng_seed=0,
        iterate_rng_seed=False,
        mix_rank_in_seed=False,
        stochastic_round_scale=False,
    )
    qparams = _eden_qparams()
    result = op.quantize(
        x, qparams, return_identity=not transpose, return_transpose=transpose,
        reduce_amax=False,
    )
    group = result.transpose if transpose else result.identity
    want, _ = op.dequantize(qtensor_group=group, qparams=qparams, is_data_t=transpose)
    for backend in (eden_ref, eden_ct):
        got = backend.quant_dequant(x, transpose=transpose, use_sr=False, seed=0)
        assert torch.equal(
            got.contiguous().view(torch.int16), want.contiguous().view(torch.int16)
        ), backend.__name__


def _hadamard_op_100483():
    """The 100483 op, with its Hadamard matrices materialized."""
    from kitchen.config import get_qlinear_params_from_qat_params

    qlp = get_qlinear_params_from_qat_params(100483)
    op = qlp.quantize_op
    op.initialize_hadamard_matrices(
        torch.zeros(128, 128, device="cuda", dtype=torch.bfloat16),
        qlp.g_params.tensor_type,
    )
    return qlp, op


@pytest.mark.parametrize("shape", [(256, 512), (384, 256)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_rht_128_matches_kitchen(shape, transpose, inverse):
    """The dim-128 rotation on both lanes, forward and inverse, bit for bit.

    The identity (dgrad) lane is new here: 9004 never rotated one, so this is the
    first test of ``rht.transform(..., transpose=False)`` -- the ``x @ H`` form,
    mixing 128 columns -- against kitchen.
    """
    _, op = _hadamard_op_100483()
    gemm_type = op.get_gemm_types_for_tensor(quantization.TensorType.G)[
        1 if transpose else 0
    ]
    lane = gemm_type.value
    forward, inv = rht.transform_matrices(
        rht.sign_vector(torch.device("cuda"), dim=128, lane=lane)
    )
    assert torch.equal(
        op.hadamard_param_dict[gemm_type]["random_hadamard_matrix"], forward
    )
    assert torch.equal(
        op.hadamard_param_dict[gemm_type]["inverse_random_hadamard_matrix"], inv
    )

    g = torch.Generator(device="cuda").manual_seed(8)
    x = (torch.randn(shape, generator=g, device="cuda") * 3).to(torch.bfloat16)
    x = rht.pad_rows(x, 128) if transpose else rht.pad_cols(x, 128)
    want = op.perform_random_hadamard_transform_ref(
        x, gemm_type, transpose, inverse_transform=inverse
    )
    got = rht.transform(x, inv if inverse else forward, transpose=transpose)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_100483_rotates_exactly_the_lanes_the_recipe_table_claims():
    """Pin ``rht_gemms`` and ``rht_dim`` for 100483: wgrad *and* dgrad, at 128.

    The discriminating part is the dgrad entry. Under 9004 no identity lane
    rotates, so a table that only ever modelled transpose lanes was
    indistinguishable from a correct one; under 100483 G's identity lane does
    rotate, and it draws the dgrad sign vector rather than the wgrad one.
    """
    _, op = _hadamard_op_100483()
    recipe = analyze_rank_bias.RECIPES["100483"]
    assert recipe.rht_dim == 128
    assert recipe.rht_gemms == frozenset({"wgrad", "dgrad"})
    for tensor_type in quantization.TensorType:
        if tensor_type.name not in ("X", "W", "G"):
            continue
        enable_identity, enable_transpose = op.get_rht_settings_for_tensor(
            tensor_type
        )[:2]
        for transpose, enabled in ((False, enable_identity), (True, enable_transpose)):
            gemm = analyze_rank_bias.GEMM_TYPES[(tensor_type.name, transpose)]
            assert (gemm in recipe.rht_gemms) == enabled, (tensor_type, transpose)
    # G's two lanes are different GEMMs, so different checked-in sign vectors.
    identity_gemm, transpose_gemm = op.get_gemm_types_for_tensor(
        quantization.TensorType.G
    )
    assert identity_gemm != transpose_gemm
    assert not torch.equal(
        rht.sign_vector(torch.device("cuda"), dim=128, lane=identity_gemm.value),
        rht.sign_vector(torch.device("cuda"), dim=128, lane=transpose_gemm.value),
    )


@pytest.mark.parametrize("transpose", [False, True])
def test_full_100483_leaf_matches_kitchen(transpose):
    """Rotate, quantize, inverse-rotate, crop -- the whole lane, bit for bit.

    ``leaf_qdq`` is what the sweep actually calls, so this is the test that says
    the composition (which axis rotates, and that the crop happens after the
    inverse) is right, not just the leaf quantizer.
    """
    _, op = _hadamard_op_100483()
    gemm_type = op.get_gemm_types_for_tensor(quantization.TensorType.G)[
        1 if transpose else 0
    ]
    g = torch.Generator(device="cuda").manual_seed(12)
    x = (torch.randn((256, 512), generator=g, device="cuda") * 3).to(torch.bfloat16)

    padded = rht.pad_rows(x, 128) if transpose else rht.pad_cols(x, 128)
    rotated = op.perform_random_hadamard_transform_ref(padded, gemm_type, transpose)
    quantized = _kitchen_eden_qdq(rotated, transpose=transpose, seed=SEED)
    restored = op.perform_random_hadamard_transform_ref(
        quantized, gemm_type, transpose, inverse_transform=True
    )
    want = restored[: x.shape[0], : x.shape[1]].contiguous()

    matrices = rht.transform_matrices(
        rht.sign_vector(x.device, dim=128, lane=gemm_type.value)
    )
    for backend in (eden_ref, eden_ct):
        got = analyze_rank_bias.leaf_qdq(
            backend,
            x,
            transpose=transpose,
            use_sr=True,
            matrices=matrices,
            rht_dim=128,
            seed=SEED,
        )
        assert torch.equal(
            got.contiguous().view(torch.int16), want.view(torch.int16)
        ), backend.__name__


def test_sign_lifetime_matches_enable_online_randomization():
    """``Recipe.dynamic_signs`` must track kitchen's ``enable_online_randomization``.

    This decides whether a sweep re-draws the RHT sign vector per trial, and it
    is not cosmetic: with the sign frozen, 100483's MSE-vs-trials slope is ~-0.02
    instead of ~-1.0, because MS-EDEN's FP4 codes are RNE and the rotation is the
    only thing left that can decorrelate a trial. Getting this backwards for
    either recipe silently produces the wrong headline curve.
    """
    from kitchen.config import get_qlinear_params_from_qat_params

    for recipe_id in ("9004", "100483"):
        qlp = get_qlinear_params_from_qat_params(int(recipe_id))
        want = bool(qlp.hadamard_params.enable_online_randomization)
        assert analyze_rank_bias.RECIPES[recipe_id].dynamic_signs == want, recipe_id
    # The concrete claim, so a config change cannot make the loop vacuous.
    assert not analyze_rank_bias.RECIPES["9004"].dynamic_signs
    assert analyze_rank_bias.RECIPES["100483"].dynamic_signs


def test_recipe_100483_hadamard_dimension_is_128_on_both_backward_gemms():
    """``rht_dim`` and the two enabled GEMMs, straight off the recipe's config."""
    from kitchen.config import get_qlinear_params_from_qat_params

    hp = get_qlinear_params_from_qat_params(100483).hadamard_params
    assert not hp.perform_hadamard_transform_fprop
    assert hp.perform_hadamard_transform_wgrad
    assert hp.perform_hadamard_transform_dgrad
    assert hp.hadamard_dim_wgrad == hp.hadamard_dim_dgrad == 128
    assert analyze_rank_bias.RECIPES["100483"].rht_dim == 128
