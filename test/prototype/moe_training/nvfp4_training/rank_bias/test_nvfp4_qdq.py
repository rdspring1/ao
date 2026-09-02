"""Bitwise-equivalence tests: CuTe DSL kernels vs PyTorch oracle vs kitchen math.

Run with ``pytest -q test_nvfp4_qdq.py`` on any CUDA machine; nothing here needs
kitchen, psx_formats or TransformerEngine.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import pytest
import torch
from cutlass.cute.runtime import from_dlpack

from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_cutedsl as eden_ct
from torchao.prototype.moe_training.nvfp4_training.rank_bias import eden_reference as eden_ref
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_cutedsl as ct
from torchao.prototype.moe_training.nvfp4_training.rank_bias import nvfp4_reference as ref
from torchao.prototype.moe_training.nvfp4_training.rank_bias.analyze_rank_bias import make_rank_labels

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required"
)

SHAPES = [
    (64, 64),  # no padding on either path
    (130, 48),  # pads rows (32/128) and cols (64) on both paths
    (1, 16),  # single block
    (512, 1024),  # large enough to hit rounding boundaries
]


def _bits(t: torch.Tensor) -> torch.Tensor:
    dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype]
    return t.contiguous().view(dtype)


def _sample(shape, seed=0, scale=3.0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (
        torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * scale
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# CuTe DSL == PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_qdq_bitwise_matches_reference(shape, transpose, use_sr):
    x = _sample(shape)
    got = ct.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    want = ref.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_quantize_intermediates_bitwise_match(transpose, use_sr):
    """Codes, block scales and the global scale must match, not just the product."""
    x = _sample((512, 1024), seed=5)
    got = ct.quantize(x, transpose=transpose, use_sr=use_sr, seed=3)
    want = ref.quantize(x, transpose=transpose, use_sr=use_sr, seed=3)
    assert torch.equal(_bits(got.data_q), _bits(want.data_q))
    assert torch.equal(_bits(got.block_descale), _bits(want.block_descale))
    assert torch.equal(_bits(got.global_descale), _bits(want.global_descale))


def test_sr_seeds_decorrelate_and_average_out():
    """SR must vary per seed and its mean must converge on the input."""
    x = _sample((256, 256), seed=9)
    a = ct.quant_dequant(x, use_sr=True, seed=1)
    b = ct.quant_dequant(x, use_sr=True, seed=2)
    assert not torch.equal(a, b)
    mean = torch.zeros_like(x, dtype=torch.float32)
    for seed in range(64):
        mean += ct.quant_dequant(x, use_sr=True, seed=seed).float()
    mean /= 64
    one_trial_err = (a.float() - x.float()).square().mean()
    assert (mean - x.float()).square().mean() < one_trial_err / 8


def test_degenerate_blocks():
    """Zeros (incl. -0.0), a zero block, and a block that saturates FP4."""
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = torch.tensor([0.0, -0.0] * 8, device="cuda", dtype=torch.bfloat16)
    x[1, :16] = 1e3  # amax block: every element saturates to the FP4 max
    x[2, 0] = 1e-4  # tiny value against a huge tensor amax -> block scale flushes
    for transpose in (False, True):
        for use_sr in (False, True):
            got = ct.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=4)
            want = ref.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=4)
            assert torch.equal(_bits(got), _bits(want)), (transpose, use_sr)


def test_all_zero_tensor():
    """amax == 0 makes the global scale overflow; kitchen falls back to unit scale."""
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    for transpose in (False, True):
        got = ct.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        want = ref.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        assert torch.equal(_bits(got), _bits(want))
        assert torch.equal(got.float(), torch.zeros_like(got, dtype=torch.float32))


# ---------------------------------------------------------------------------
# E4M3 block-scale rounding == torch's float8_e4m3fn cast
# ---------------------------------------------------------------------------


@cute.kernel
def _e4m3_probe_kernel(gX: cute.Tensor, gO: cute.Tensor, n: cutlass.Int32):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    i = bidx * 128 + tidx
    if i < n:
        gO[i] = ct._e4m3_rne(cutlass.Float32(gX[i]))


@cute.jit
def _e4m3_probe(mX: cute.Tensor, mO: cute.Tensor, n: cutlass.Int32, stream):
    _e4m3_probe_kernel(mX, mO, n).launch(
        grid=[(n + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
    )


def test_e4m3_rne_matches_torch_cast_exhaustively():
    """Every fp32 bit pattern in [0, 449) plus the saturating tail."""
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    chunk = 1 << 22
    for start in range(0, 0x43E10000, chunk):
        pattern = torch.arange(
            start, min(start + chunk, 0x43E10000), device="cuda", dtype=torch.int64
        ).to(torch.int32)
        vals = pattern.view(torch.float32)
        out = torch.empty_like(vals)
        _e4m3_probe(from_dlpack(vals), from_dlpack(out), vals.numel(), stream)
        want = vals.to(torch.float8_e4m3fn).float()
        assert torch.equal(_bits(out), _bits(want)), f"chunk at {start:#x}"

    tail = torch.tensor(
        [448.0, 448.001, 449.0, 464.0, 1e10, float("inf")], device="cuda"
    )
    out = torch.empty_like(tail)
    _e4m3_probe(from_dlpack(tail), from_dlpack(out), tail.numel(), stream)
    assert torch.equal(_bits(out), _bits(tail.to(torch.float8_e4m3fn).float()))


# ---------------------------------------------------------------------------
# PyTorch reference == kitchen's nvfp_utils / cast_utils source
# ---------------------------------------------------------------------------


def _kitchen_cast_to_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Verbatim kitchen/cast_utils.py::cast_to_fp4_e2m1."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def _kitchen_cast_to_fp4_e2m1_sr(x: torch.Tensor, sampled_prob: torch.Tensor):
    """kitchen/cast_utils.py::cast_to_fp4_e2m1_sr; only its ``torch.rand`` call
    is replaced by the injected uniforms."""
    sign = torch.sign(x)
    x = torch.abs(x)
    x_fp4_high = torch.where(
        x > 4,
        6,
        torch.where(
            x > 3,
            4,
            torch.where(
                x > 2,
                3,
                torch.where(
                    x > 1.5, 2, torch.where(x > 1.0, 1.5, torch.where(x > 0.5, 1, 0.5))
                ),
            ),
        ),
    )
    x_fp4_low = torch.where(
        x > 4,
        4,
        torch.where(
            x > 3,
            3,
            torch.where(
                x > 2,
                2,
                torch.where(
                    x > 1.5,
                    1.5,
                    torch.where(x > 1.0, 1.0, torch.where(x > 0.5, 0.5, 0.0)),
                ),
            ),
        ),
    )
    prob_up = (x - x_fp4_low) / (x_fp4_high - x_fp4_low)
    return torch.where(sampled_prob < prob_up, x_fp4_high, x_fp4_low) * sign


def _kitchen_qdq(data_hp: torch.Tensor, use_sr: bool, uniforms=None) -> torch.Tensor:
    """kitchen/nvfp_utils.py::to_nvfp_verbose + from_nvfp_verbose, transcribed for
    quant_tile_shape=(1, 16), E2M1, E4M3_RNE scales, per-tensor 2-level scaling."""
    M, N = data_hp.shape
    blk_m, blk_n, block_size0, block_size1 = M, N // 16, 1, 16
    amax_encoded_data, amax_encoded_block_scale = 6.0, 448.0

    data_blocked = (
        data_hp.reshape(blk_m, block_size0, blk_n, block_size1)
        .permute(0, 2, 1, 3)
        .contiguous()
        .flatten(start_dim=2)
    )
    max_abs_per_block = torch.amax(
        torch.abs(data_blocked), dim=-1, keepdim=True
    ).float()

    global_amax = data_hp.abs().float().amax().reshape(1)  # compute_tensor_absmax
    global_scaling_factor = torch.div(
        amax_encoded_data * amax_encoded_block_scale, global_amax
    )
    global_descaling_factor = torch.reciprocal(global_scaling_factor)
    overflow_mask = torch.isinf(global_scaling_factor)
    if torch.any(overflow_mask):
        ones = torch.ones_like(global_scaling_factor)
        global_scaling_factor = torch.where(overflow_mask, ones, global_scaling_factor)
        global_descaling_factor = torch.where(
            overflow_mask, ones, global_descaling_factor
        )
    global_scaling_factor = global_scaling_factor.expand(blk_m, blk_n, 1)
    global_descaling_factor = global_descaling_factor.expand(blk_m, blk_n, 1)

    block_descaling_factor_e4m3 = (
        torch.div(max_abs_per_block, amax_encoded_data) * global_scaling_factor
    ).to(torch.float8_e4m3fn)
    block_scaling_factor = torch.where(
        block_descaling_factor_e4m3 == 0,
        1.0,
        torch.reciprocal(block_descaling_factor_e4m3.float() * global_descaling_factor),
    )

    data_hp_q = data_blocked * block_scaling_factor
    if use_sr:
        data_hp_q = _kitchen_cast_to_fp4_e2m1_sr(
            data_hp_q, uniforms.reshape(data_hp_q.shape)
        )
    else:
        data_hp_q = _kitchen_cast_to_fp4_e2m1(data_hp_q)

    data_hp_q = (
        data_hp_q.reshape(blk_m, blk_n, block_size0, block_size1)
        .permute(0, 2, 1, 3)
        .reshape(M, N)
        .float()
    )
    block_descale = block_descaling_factor_e4m3.squeeze(-1)
    global_descale = global_descaling_factor.squeeze(-1)

    # from_nvfp_verbose
    restored = (
        data_hp_q
        * block_descale.float()
        .repeat_interleave(block_size0, 0)
        .repeat_interleave(block_size1, 1)
        * global_descale.repeat_interleave(block_size0, 0).repeat_interleave(
            block_size1, 1
        )
    )
    return restored.to(torch.bfloat16)  # QuantizeOpBase.dequantize


@pytest.mark.parametrize("use_sr", [False, True])
def test_reference_matches_kitchen_source(use_sr):
    """The oracle reproduces kitchen's own reference math bit for bit."""
    x = _sample((256, 512), seed=7)
    uniforms = ref.philox_uniform(x.numel(), 21, x.device) if use_sr else None
    want = _kitchen_qdq(x, use_sr, uniforms)
    got = ref.quant_dequant(x, use_sr=use_sr, seed=21)
    assert torch.equal(_bits(got), _bits(want))


# ---------------------------------------------------------------------------
# Rank labelling used by the sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transpose", [False, True])
def test_rank_labels_follow_the_quantization_block_axis(transpose):
    """Identity ranks within a row (1xB); transpose ranks within a column (Bx1)."""
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 0.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float32
    )
    labels = make_rank_labels(x, block_size=2, transpose=transpose)
    expected = (
        [[2, 1], [1, 0], [2, 2], [1, 1]]
        if transpose
        else [[2, 1], [1, 0], [2, 1], [2, 1]]
    )
    assert labels.tolist() == expected
    assert (labels == 0).equal(x == 0)


# ---------------------------------------------------------------------------
# MS-EDEN (recipe 100483): CuTe DSL == PyTorch oracle, and the pieces kitchen
# models itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", SHAPES + [(300, 700)])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_eden_qdq_bitwise_matches_reference(shape, transpose, use_sr):
    x = _sample(shape)
    got = eden_ct.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    want = eden_ref.quant_dequant(x, transpose=transpose, use_sr=use_sr, seed=11)
    assert torch.equal(_bits(got), _bits(want))


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("use_sr", [False, True])
def test_eden_intermediates_bitwise_match(transpose, use_sr):
    """Codes, corrected block scales and the global scale, not just the product."""
    x = _sample((512, 1024), seed=5)
    got = eden_ct.quantize(
        x, transpose=transpose, seed=3, stochastic_round_scale=use_sr
    )
    want = eden_ref.quantize(
        x, transpose=transpose, seed=3, stochastic_round_scale=use_sr
    )
    assert torch.equal(_bits(got.data_q), _bits(want.data_q))
    assert torch.equal(_bits(got.block_descale), _bits(want.block_descale))
    assert torch.equal(_bits(got.global_descale), _bits(want.global_descale))


def test_eden_degenerate_blocks():
    """The same degenerate blocks the 9004 path is pinned on, on the Eden path.

    Eden reaches them differently: a zero block scale clamps the encode scale to
    FLT_MAX instead of falling back to unit scaling, and an all-zero block makes
    ``sum_cross`` zero so the correction has to fall back to 1 rather than divide
    by zero.
    """
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    x[0, :16] = torch.tensor([0.0, -0.0] * 8, device="cuda", dtype=torch.bfloat16)
    x[1, :16] = 1e3
    x[2, 0] = 1e-4
    for transpose in (False, True):
        for use_sr in (False, True):
            got = eden_ct.quant_dequant(
                x, transpose=transpose, use_sr=use_sr, seed=4
            )
            want = eden_ref.quant_dequant(
                x, transpose=transpose, use_sr=use_sr, seed=4
            )
            assert torch.equal(_bits(got), _bits(want)), (transpose, use_sr)
            assert torch.isfinite(got.float()).all()


def test_eden_all_zero_tensor():
    x = torch.zeros(32, 32, device="cuda", dtype=torch.bfloat16)
    for transpose in (False, True):
        got = eden_ct.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        want = eden_ref.quant_dequant(x, transpose=transpose, use_sr=True, seed=1)
        assert torch.equal(_bits(got), _bits(want))
        assert torch.equal(got.float(), torch.zeros_like(got, dtype=torch.float32))


def test_eden_correction_is_one_on_an_exactly_representable_block():
    """A block already on the FP4 grid needs no correction, and round-trips exactly.

    This is the test that says the correction is a *correction* and not a
    rescale: ``sum_sq == sum_cross`` when every code equals its input, so the
    ratio is exactly 1.0 and the scale is unchanged.
    """
    grid = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0,
         -4.0, -6.0, 0.0],
        device="cuda",
    )
    x = grid.repeat(16, 1).to(torch.bfloat16)
    q = eden_ref.quantize(x, seed=0, stochastic_round_scale=False)
    # amax 6 and numerator 1536 put the block scale on an exact E4M3 value, so
    # an unchanged correction means an exact round trip.
    assert torch.equal(
        eden_ref.quant_dequant(x, use_sr=False, seed=0).float(), x.float()
    )
    assert torch.isfinite(q.block_descale).all()


def test_eden_correction_falls_back_to_one_when_every_code_is_zero():
    """``sum_cross == 0`` must clamp the correction, not divide by zero.

    Values below the FP4 rounding threshold against a much larger tensor amax
    quantize to all zeros, which is the only way the denominator vanishes.
    """
    x = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1e3  # sets the tensor amax
    x[1, 16:] = 1e-3  # a block whose codes all round to zero
    got = eden_ref.quant_dequant(x, use_sr=False, seed=0)
    assert torch.isfinite(got.float()).all()
    assert torch.equal(
        got[1, 16:].float(), torch.zeros(16, device="cuda", dtype=torch.float32)
    )
    assert torch.equal(_bits(eden_ct.quant_dequant(x, use_sr=False, seed=0)), _bits(got))


def test_eden_block_scale_never_exceeds_the_256_ceiling():
    """The Eden numerator is 6*256, leaving headroom for the correction to grow
    the scale without saturating E4M3 at 448."""
    x = _sample((256, 256), seed=13, scale=50.0)
    q = eden_ref.quantize(x, seed=1)
    assert q.block_descale.max().item() <= 448.0
    uncorrected = eden_ref.quantize(x, seed=1, stochastic_round_scale=False)
    assert uncorrected.block_descale.max().item() <= eden_ref.EDEN_BLOCK_SCALE_MAX


def test_eden_data_is_deterministic_and_only_the_scale_is_random():
    """MS-EDEN's data codes are RNE, so two seeds differ only in the block scale.

    This is the structural difference from 9004 that the whole recipe turns on,
    and it is also why a trial mean converges on the *corrected* reconstruction
    rather than on the input.
    """
    x = _sample((256, 256), seed=17)
    a = eden_ref.quantize(x, seed=1)
    b = eden_ref.quantize(x, seed=2)
    assert torch.equal(_bits(a.data_q), _bits(b.data_q))
    assert not torch.equal(_bits(a.block_descale), _bits(b.block_descale))


def test_eden_scale_sr_is_unbiased():
    """``emulate_sr_e4m3`` must average back to the value it was given.

    Stated on the primitive rather than on the block scale, because SR is
    unbiased with respect to the *unrounded* corrected scale; the quantizer only
    ever exposes it already rounded, so a block-level mean is off by the RNE
    error of the deterministic branch and would not test SR at all.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    values = torch.rand(4096, generator=g, device="cuda") * 200.0
    trials = 4096
    mean = torch.zeros_like(values, dtype=torch.float64)
    for t in range(trials):
        rbits = torch.randint(
            0, 2**32, values.shape, generator=g, device="cuda", dtype=torch.int64
        )
        mean += eden_ref.emulate_sr_e4m3(values, rbits).double()
    mean /= trials
    rel = ((mean - values.double()).abs() / values.double()).max().item()
    # One E4M3 step is 2**-3 relative; the standard error over `trials` draws is
    # far below that, so a systematic bias would show here immediately.
    assert rel < 0.01, rel


def test_eden_row_and_column_lanes_draw_independent_streams():
    """The two lanes key on different Philox subsequences, so a block that covers
    the same elements on both paths must not reuse the same random word."""
    seq_i, word_i = eden_ref.block_subsequence(256, 256, False, torch.device("cuda"))
    seq_t, word_t = eden_ref.block_subsequence(256, 256, True, torch.device("cuda"))
    assert not torch.equal(seq_i, seq_t)
    rb_i = eden_ref.block_rbits(256, 256, False, 5, torch.device("cuda"))
    rb_t = eden_ref.block_rbits(256, 256, True, 5, torch.device("cuda"))
    assert not torch.equal(rb_i, rb_t)


def test_eden_fma_helper_matches_a_real_gpu_fma():
    """``_fma`` is what makes the oracle's correction bitwise, so pin it.

    A plain ``a * b + c`` in torch rounds twice and disagrees with the kernel's
    ``fma.rn.f32`` on a large fraction of random inputs; the round-to-odd
    narrowing in ``_fma`` is what removes that.
    """
    g = torch.Generator(device="cuda").manual_seed(0)
    a, b, c = (torch.randn(1 << 16, device="cuda", generator=g) for _ in range(3))
    fused = eden_ref._fma(a, b, c)
    naive = a * b + c
    assert not torch.equal(_bits(fused), _bits(naive)), "test would be vacuous"
    # torch.addcmul is documented as unfused, so it is not an oracle; instead
    # check the defining property: the result is the correctly rounded fp32 of
    # the exact a*b+c, computed in float64 where a*b is exact.
    exact = a.double() * b.double() + c.double()
    off_by_more_than_one_ulp = (
        (fused.double() - exact).abs() > (exact.abs() * 2.0**-23)
    ).sum()
    assert off_by_more_than_one_ulp.item() == 0
