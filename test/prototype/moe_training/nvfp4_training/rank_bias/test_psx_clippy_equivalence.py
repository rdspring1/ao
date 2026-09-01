"""Cross-check against the psx ``clippy`` CUDA kernels kitchen actually dispatches to.

Needs ``psx_formats`` (kitchen's ``third_party/psx-formats``) built; skipped
otherwise. Nothing here needs kitchen itself: for recipe 6302's 1x16 E2M1
configuration ``QuantizeOpNVFP4Emulation`` reduces to pad -> ``nvfp4_clippy`` /
``nvfp4_clippy_transpose`` (``scale_rounding_mode=E4M3_RNE``) -> trim, which is
what :func:`clippy_qdq` spells out. ``test_kitchen_equivalence.py`` drives the
same kernels through the real op, and needs kitchen built.

The RNE path is bitwise. The SR path cannot be: clippy's stochastic rounding
draws from ``curand`` inside the kernel and rounds by adding random bits to the
mantissa, so only the *stream* is comparable, never the bits. What is checked
instead is that every clippy SR output lands on one of the two FP4 grid points
this package's SR chooses between -- i.e. identical scales and identical
candidates -- and that the trial mean converges on the input.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("torch.cuda")
psx = pytest.importorskip(
    "psx_formats.utils", reason="psx_formats is not built in this environment"
)

from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_cutedsl as ct  # noqa: E402
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_reference as ref  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

E4M3_RNE = 0  # kitchen.quantization.ClippyScaleRoundingMode.E4M3_RNE


def clippy_qdq(x: torch.Tensor, *, transpose: bool = False, use_sr: bool = False):
    """``QuantizeOpNVFP4Emulation``'s leaf path, called straight into psx."""
    padded = ref.pad_2d(x, *(ref.PAD_TRANSPOSE if transpose else ref.PAD_IDENTITY))
    padded = padded.contiguous()
    amax = padded.abs().float().amax().reshape(1)  # ops.compute_tensor_absmax
    kernel = psx.nvfp4_clippy_transpose if transpose else psx.nvfp4_clippy
    out = kernel(
        padded,
        amax,
        use_rs=use_sr,
        scale_rounding_mode=E4M3_RNE,
        sat_count=torch.zeros(1, dtype=torch.int32, device=x.device),
    )
    return out[: x.shape[0], : x.shape[1]].contiguous()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _sample(shape, seed=3, scale=3.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(shape, generator=g, device="cuda") * scale).to(torch.bfloat16)


@pytest.mark.parametrize("shape", [(256, 512), (130, 48), (1024, 1024)])
@pytest.mark.parametrize("transpose", [False, True])
def test_rne_matches_clippy_bitwise(shape, transpose):
    x = _sample(shape)
    want = clippy_qdq(x, transpose=transpose)
    assert torch.equal(_bits(ref.quant_dequant(x, transpose=transpose)), _bits(want))
    assert torch.equal(_bits(ct.quant_dequant(x, transpose=transpose)), _bits(want))


@pytest.mark.parametrize("transpose", [False, True])
def test_rne_matches_clippy_across_shapes_and_magnitudes(transpose):
    """Sweep the regime a real dump lives in: 1e-4..1e4, ragged shapes, zeros."""
    for trial in range(12):
        g = torch.Generator(device="cuda").manual_seed(trial)
        rows = int(torch.randint(1, 300, (1,), generator=g, device="cuda").item())
        cols = 16 * int(torch.randint(1, 40, (1,), generator=g, device="cuda").item())
        exponent = int(torch.randint(-4, 5, (1,), generator=g, device="cuda").item())
        x = (torch.randn((rows, cols), generator=g, device="cuda") * 10.0**exponent).to(
            torch.bfloat16
        )
        zeros = torch.rand((rows, cols), generator=g, device="cuda") < 0.1
        x = x.masked_fill(zeros, 0.0)
        got = ref.quant_dequant(x, transpose=transpose)
        assert torch.equal(_bits(got), _bits(clippy_qdq(x, transpose=transpose))), (
            f"trial {trial}: shape {(rows, cols)}, magnitude 1e{exponent}"
        )


@pytest.mark.parametrize("transpose", [False, True])
def test_sr_shares_clippy_scales_and_candidates(transpose, monkeypatch):
    """Clippy's SR output must land on the two grid points our SR chooses between.

    Forcing the uniform to 0 / 1-ulp collapses this package's SR onto the upper
    and lower FP4 neighbour, so the pair brackets every stochastic outcome. Same
    bracket for every element means same block scale, same global scale and same
    FP4 grid; only the random stream differs.
    """
    x = _sample((256, 512), seed=5)

    def bracket(u_const: float) -> torch.Tensor:
        monkeypatch.setattr(
            ref,
            "philox_uniform",
            lambda numel, seed, device: torch.full((numel,), u_const, device=device),
        )
        try:
            return ref.quant_dequant(x, transpose=transpose, use_sr=True).float()
        finally:
            monkeypatch.undo()

    high, low = bracket(0.0), bracket(1.0 - 2**-24)
    trials = 32
    total = torch.zeros_like(high)
    for _ in range(trials):
        out = clippy_qdq(x, transpose=transpose, use_sr=True).float()
        assert bool(((out == high) | (out == low)).all())
        total += out
    assert not torch.equal(total / trials, high), "clippy SR never rounded down"
    # SR is unbiased, so the trial mean converges on the input.
    err = (total / trials - x.float()).abs().mean()
    assert err < 0.1 * x.float().abs().mean(), err


def test_known_divergences_from_clippy():
    """Two extreme-dynamic-range regimes where clippy and kitchen's own reference
    math disagree, plus signed zero. All are outside the regime of a real dump;
    see README.md. This test pins them so a change in either side is noticed."""
    # 1. -0.0 input: clippy carries the sign through copysignf, kitchen's
    #    cast_utils multiplies by torch.sign(x), which is 0 for +-0. Same value,
    #    different sign bit.
    x = torch.zeros(32, 16, dtype=torch.bfloat16, device="cuda")
    x[:, 0] = 6.0
    x[:, 1] = -0.0
    got, want = ref.quant_dequant(x), clippy_qdq(x)
    assert torch.equal(got.float(), want.float())
    assert not torch.equal(_bits(got), _bits(want))
    assert (_bits(want)[:, 1] == -32768).all()  # clippy keeps -0.0
    assert (_bits(got)[:, 1] == 0).all()  # reference returns +0.0

    # 2. global_amax * block_amax overflows fp32 (needs an amax around 1e19+):
    #    clippy's combined-amax guard zeroes the whole block, the reference
    #    reconstructs it.
    x = torch.full((32, 16), 1e20, dtype=torch.bfloat16, device="cuda")
    assert (clippy_qdq(x) == 0).all()
    assert (ref.quant_dequant(x) != 0).all()

    # 3. E4M3 block scale underflows to zero (block amax < 2**-9 / 448 of the
    #    tensor amax) while the block still holds values above the FP4 rounding
    #    threshold: clippy substitutes a unit block scale and reconstructs
    #    code * global_decode_scale, the reference dequantizes through the zero
    #    scale and returns 0.
    x = torch.ones((32, 16), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = 1e18
    assert (ref.quant_dequant(x)[1:] == 0).all()
    assert (clippy_qdq(x)[1:] != 0).all()
